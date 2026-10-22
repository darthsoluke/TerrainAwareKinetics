"""Compare native model losses and parameter gradients for exact short convs."""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from motionbricks.data.full_terrain_cache import PrimitiveDataset,collate_primitives
from motionbricks.data.short_sequence_convolution import short_sequence_convolutions,VERSION
from motionbricks.data.unified_full_training import NativeTrainingStage
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_unified_full_corpus import device_batch
from scripts.train_joint_terrain_v3 import write_json


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    inference,bundle=load_terrain_bundle(a.bundle,'cuda:0')
    controller=UnifiedTerrainController(inference,bundle)
    saved=torch.load(a.initial,map_location='cpu',weights_only=False)
    dataset=PrimitiveDataset(a.fixture/'data','train')
    ids={dataset.records[actual]['domain']:i for i,actual in enumerate(dataset.ids)}
    batch=device_batch(collate_primitives([dataset[ids[domain]] for domain in ('pfnn','pfnn','parc','bones')]),'cuda:0')
    rows=[]
    for name in ('root','codec','pose'):
        torch.use_deterministic_algorithms(True,warn_only=name=='codec')
        expected=None
        for optimized in (False,True):
            controller.load_state_dict(saved['model_state'],strict=True)
            controller.zero_grad(set_to_none=True)
            stage=NativeTrainingStage(controller,name).train(True)
            torch.manual_seed(2333)
            with short_sequence_convolutions(controller) if optimized else nullcontext():
                loss=stage(batch,2333)
                loss.backward()
            gradients={key:p.grad.detach().cpu().clone() if p.grad is not None else None
                for key,p in controller.named_parameters() if p.requires_grad}
            if not optimized:
                expected,reference_loss=gradients,float(loss.detach())
            else:
                absolute=0.
                squared_error=squared_reference=0.
                for key,ref in expected.items():
                    actual=gradients[key]
                    if ref is None or actual is None:
                        if ref is not None or actual is not None:
                            raise ValueError('Gradient presence differs: '+key)
                        continue
                    torch.testing.assert_close(actual,ref,atol=5e-6,rtol=1e-3)
                    absolute=max(absolute,float((actual-ref).abs().max()))
                    squared_error+=float((actual.double()-ref.double()).square().sum())
                    squared_reference+=float(ref.double().square().sum())
                torch.testing.assert_close(loss.detach().cpu(),torch.tensor(reference_loss),atol=1e-6,rtol=1e-5)
                rows.append(dict(stage=name,reference_loss=reference_loss,optimized_loss=float(loss.detach()),
                    gradient_max_abs=absolute,gradient_relative_l2=(squared_error/max(squared_reference,1e-30))**.5))
                write_json(a.output/'progress.json',dict(rows=rows))
        del expected,gradients,loss,stage
    controller.close()
    write_json(a.output/'complete.json',dict(passed=True,version=VERSION,rows=rows,
        actual_optimizer_updates=0,model_exported=False,batch=4))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','initial','fixture','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json',dict(error=repr(error)))
        raise
