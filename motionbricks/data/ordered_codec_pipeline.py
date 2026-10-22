"""Ordered multi-device codec computation with serial native EMA transitions."""

from concurrent.futures import ThreadPoolExecutor
from threading import Condition

import torch

from .parallel_root_training import to_device
from .exposure_codec_training import ordered_ema_chunks, quantize_in_chunks


class OrderedCodecPipeline:
    def __init__(self, stages, micro_batch=64, micro_count=4):
        if len(stages) not in (2,4) or any(stage.stage!='codec' or not stage.graph_dilated_convolutions for stage in stages):
            raise ValueError('Pipeline requires two or four graph-enabled codec replicas')
        self.stages=stages
        if micro_batch not in (64,128,256,512,1024) or micro_count not in (2,4) or micro_count%len(stages):
            raise ValueError('Unsupported codec compute batch configuration')
        self.micro_batch,self.micro_count=micro_batch,micro_count
        self.controllers=[stage.controller for stage in stages]
        self.parameters=[{name:p for name,p in controller.named_parameters() if p.requires_grad}
            for controller in self.controllers]
        if any(self.parameters[0].keys()!=parameters.keys() for parameters in self.parameters[1:]):raise ValueError('Replica scopes differ')
        self.devices=[next(iter(parameters.values())).device for parameters in self.parameters]
        if len(set(self.devices))!=len(stages) or any(device.type!='cuda' for device in self.devices):
            raise ValueError('Distinct CUDA devices required')
        self.pools=[ThreadPoolExecutor(max_workers=1) for _ in stages]
        self.ready=False
        self.calibration_microbatches=0
        self.last_outputs=[]
        self.last_ema_order=[]
        self.warm_memory=[]

    def close(self):
        for pool in self.pools:pool.shutdown(wait=True)

    def evaluation_stage(self):
        with torch.no_grad(),torch.cuda.device(self.devices[1]):
            self.controllers[1].load_state_dict(self.controllers[0].state_dict(),strict=True)
        for device in self.devices:torch.cuda.synchronize(device)
        return self.stages[1]

    def validate(self,function,*args):
        stage=self.evaluation_stage()
        previous=stage.graph_dilated_convolutions
        stage.graph_dilated_convolutions=False
        try:return function(stage,*args)
        finally:stage.graph_dilated_convolutions=previous

    def _warm(self, worker, microbatch, seed):
        controller=self.controllers[worker];device=self.devices[worker]
        with torch.cuda.device(device):
            # Keep the rollback copy off CUDA so capture has production-sized
            # convolution workspace available when choosing native algorithms.
            before={name:value.detach().cpu().clone() for name,value in controller.state_dict().items()}
            allocated_before=torch.cuda.memory_allocated(device)
            cpu=torch.get_rng_state();rng=torch.cuda.get_rng_state(device)
            controller.zero_grad(set_to_none=True)
            with ordered_ema_chunks(controller):
                loss=self.stages[worker](to_device(microbatch,device),seed)
                (loss/self.micro_count).backward()
            torch.cuda.synchronize(device)
            controller.load_state_dict(before,strict=True);controller.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(rng,device)
            self.calibration_microbatches+=1
            self.warm_memory.append(dict(worker=worker,allocated_before=allocated_before,
                allocated_after=torch.cuda.memory_allocated(device),reserved_after=torch.cuda.memory_reserved(device)))

    def backward(self,microbatches,seeds):
        if len(microbatches)!=self.micro_count or len(seeds)!=self.micro_count or any(len(batch['world'])!=self.micro_batch for batch in microbatches):
            raise ValueError('Keep the declared compute batches in their original order')
        if not self.ready:
            # Capture is serialized, before any cross-device work. Capturing a
            # new graph while another worker executes CUDA would be invalid.
            for worker in range(len(self.stages)):
                self.pools[worker].submit(self._warm,worker,microbatches[worker],seeds[worker]).result()
            self.ready=True
        cpu=torch.get_rng_state()
        primary=self.devices[0]
        for worker in range(1,len(self.stages)):
            with torch.no_grad(),torch.cuda.device(self.devices[worker]):
                for name,value in self.parameters[worker].items():value.copy_(self.parameters[0][name])
        for device in self.devices:torch.cuda.synchronize(device)
        condition=Condition()
        shared=dict(next=0,error=None,rng=torch.cuda.get_rng_state(primary),
            quantizer={name:value.detach().clone() for name,value in self.controllers[0].codec.quantizer.state_dict().items()})
        self.last_ema_order=[]

        def execute(index):
            worker=index%len(self.stages);device=self.devices[worker];controller=self.controllers[worker]
            quantizer=controller.codec.quantizer
            previous=quantizer.__dict__.get('forward');native=quantizer.forward
            with torch.cuda.device(device):
                controller.zero_grad(set_to_none=True)
                batch=to_device(microbatches[index],device)
                entry_rng=torch.cuda.get_rng_state(device)
                after_quantizer=[]
                def ordered(*args,**kwargs):
                    with condition:
                        condition.wait_for(lambda:shared['next']==index or shared['error'] is not None)
                        if shared['error'] is not None:raise RuntimeError('Another codec worker failed')
                        if not torch.equal(entry_rng,torch.cuda.get_rng_state(device)):
                            raise ValueError('Codec consumed RNG before the ordered quantizer')
                        with torch.no_grad():
                            quantizer.load_state_dict(shared['quantizer'],strict=True)
                        torch.cuda.set_rng_state(shared['rng'],device)
                        if kwargs or len(args)!=1:raise ValueError('Unexpected codec quantizer call')
                        output=quantize_in_chunks(native,args[0])
                        torch.cuda.synchronize(device)
                        shared['quantizer']={name:value.detach().clone() for name,value in quantizer.state_dict().items()}
                        torch.cuda.synchronize(device)
                        shared['rng']=torch.cuda.get_rng_state(device)
                        after_quantizer.append(shared['rng'].clone())
                        self.last_ema_order.append(index)
                        shared['next']+=1;condition.notify_all()
                        return output
                quantizer.forward=ordered
                try:
                    signatures=len(self.stages[worker]._dilated_graph_execution.calls)
                    loss=self.stages[worker](batch,seeds[index]);(loss/self.micro_count).backward()
                    torch.cuda.synchronize(device)
                    if len(after_quantizer)!=1 or not torch.equal(after_quantizer[0],torch.cuda.get_rng_state(device)):
                        raise ValueError('Codec RNG outside the ordered quantizer changed')
                    if len(self.stages[worker]._dilated_graph_execution.calls)!=signatures:
                        raise ValueError('Unexpected graph capture during pipeline execution')
                    # Both results are consumed before the next pair can clear
                    # gradients, so the original buffers remain valid here.
                    return float(loss.detach()),{name:None if p.grad is None else p.grad.detach()
                        for name,p in self.parameters[worker].items()},controller.last_local.detach().cpu().clone()
                except Exception as error:
                    with condition:shared['error']=repr(error);condition.notify_all()
                    raise
                finally:
                    if previous is None:del quantizer.forward
                    else:quantizer.forward=previous

        accumulated,losses,outputs={},[],[]
        for first in range(0,self.micro_count,len(self.stages)):
            futures=[self.pools[worker].submit(execute,first+worker) for worker in range(len(self.stages))]
            for future in futures:
                loss,gradients,output=future.result();losses.append(loss);outputs.append(output)
                with torch.no_grad():
                    for name,gradient in gradients.items():
                        if gradient is None:continue
                        if name not in accumulated:accumulated[name]=gradient.to(primary)
                        else:accumulated[name].add_(gradient.to(primary))
        if shared['next']!=self.micro_count or self.last_ema_order!=list(range(self.micro_count)) or not torch.equal(cpu,torch.get_rng_state()):
            raise ValueError('Codec EMA order or shared CPU RNG changed')
        with torch.no_grad():
            self.controllers[0].codec.quantizer.load_state_dict(shared['quantizer'],strict=True)
        for name,parameter in self.parameters[0].items():parameter.grad=accumulated.get(name)
        for device in self.devices:torch.cuda.set_rng_state(shared['rng'],device)
        self.last_outputs=outputs
        return losses
