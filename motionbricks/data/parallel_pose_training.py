"""Two Pose replicas sharing an ordered logical dropout stream and one Adam."""

from concurrent.futures import ThreadPoolExecutor

import torch

from .parallel_root_training import to_device


def pose_micro_weights(sizes, maximum=3072):
    if not sizes or len(sizes) > 2 or any(type(size) is not int or size <= 0
            or size > maximum or size % 256 for size in sizes):
        raise ValueError('Pose requires one or two nonempty whole-reference microbatches')
    if len(sizes) == 2 and sizes[0] != maximum:
        raise ValueError('Only the final Pose microbatch may be partial')
    total = sum(sizes)
    return [size / total for size in sizes]


class ParallelPoseMicrobatches:
    def __init__(self, stages, *, capture_outputs=False):
        if len(stages) != 2 or any(stage.stage != 'pose' or not stage.cache_pose_tokens
                or not stage.coalesce_dilated_bias or not stage.isolate_pose_teacher_rng for stage in stages):
            raise ValueError('Two cached Pose replicas with isolated teacher RNG are required')
        self.stages = stages
        self.controllers = [stage.controller for stage in stages]
        if any(controller.codec.training or any(p.requires_grad for p in controller.codec.parameters())
               for controller in self.controllers):
            raise ValueError('Pose replicas require frozen eval codecs')
        self.parameters = [{name: p for name, p in controller.named_parameters() if p.requires_grad}
                           for controller in self.controllers]
        if not self.parameters[0] or self.parameters[0].keys() != self.parameters[1].keys():
            raise ValueError('Pose replica trainable scopes differ')
        self.devices = [next(iter(parameters.values())).device for parameters in self.parameters]
        if len(set(self.devices)) != 2 or any(device.type != 'cuda' for device in self.devices):
            raise ValueError('Two distinct CUDA devices are required')
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.rng_increments = {}
        self.calibration_microbatches = 0
        self.capture_outputs = capture_outputs
        self.last_outputs = []

    def close(self):
        self.pool.shutdown(wait=True)

    def _calibrate(self, batch, seed, weight):
        primary = self.devices[0]
        controller = self.controllers[0]
        cpu = torch.get_rng_state()
        before = torch.cuda.get_rng_state(primary)
        generator = torch.cuda.default_generators[primary.index]
        offset = generator.get_offset()
        try:
            with torch.cuda.device(primary):
                controller.zero_grad(set_to_none=True)
                loss = self.stages[0](to_device(batch, primary), seed)
                (loss * weight).backward()
                torch.cuda.synchronize(primary)
            increment = generator.get_offset() - offset
            if increment <= 0 or increment % 4 or not torch.equal(cpu, torch.get_rng_state()):
                raise ValueError('Pose dropout must follow a CUDA-only Philox schedule')
            self.rng_increments[len(batch['world'])] = increment
            self.calibration_microbatches += 1
        finally:
            controller.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(before, primary)

    def backward(self, microbatches, seeds):
        sizes = [len(batch['world']) for batch in microbatches]
        weights = pose_micro_weights(sizes)
        if len(seeds) != len(microbatches):
            raise ValueError('Each original microbatch requires its own teacher seed')
        for batch, seed, weight in zip(microbatches, seeds, weights):
            if len(batch['world']) not in self.rng_increments:
                self._calibrate(batch, seed, weight)
        primary = self.devices[0]
        cpu = torch.get_rng_state()
        initial_rng = torch.cuda.get_rng_state(primary)
        initial_offset = torch.cuda.default_generators[primary.index].get_offset()
        starts, offset = [], initial_offset
        for size in sizes:
            starts.append(offset)
            offset += self.rng_increments[size]
        for worker in range(1, len(microbatches)):
            with torch.no_grad(), torch.cuda.device(self.devices[worker]):
                for name, parameter in self.parameters[worker].items():
                    parameter.copy_(self.parameters[0][name], non_blocking=True)
        for device in self.devices:
            torch.cuda.synchronize(device)

        def execute(worker):
            device, controller = self.devices[worker], self.controllers[worker]
            with torch.cuda.device(device):
                controller.zero_grad(set_to_none=True)
                torch.cuda.set_rng_state(initial_rng, device)
                generator = torch.cuda.default_generators[device.index]
                generator.set_offset(starts[worker])
                loss = self.stages[worker](to_device(microbatches[worker], device), seeds[worker])
                (loss * weights[worker]).backward()
                torch.cuda.synchronize(device)
                if generator.get_offset() != starts[worker] + self.rng_increments[sizes[worker]]:
                    raise ValueError('Pose dropout RNG increment changed; reject optimizer update')
                output = controller.last_logits.detach().cpu().clone() if self.capture_outputs else None
                return float(loss.detach()), {name: None if p.grad is None else p.grad.detach()
                    for name, p in self.parameters[worker].items()}, output

        futures = [self.pool.submit(execute, worker) for worker in range(len(microbatches))]
        accumulated, losses, outputs = {}, [], []
        with torch.no_grad(), torch.cuda.device(primary):
            # Merge in the same order as serial accumulation, never completion order.
            for future in futures:
                loss, gradients, output = future.result()
                losses.append(loss)
                outputs.append(output)
                for name, gradient in gradients.items():
                    if gradient is None:
                        continue
                    if name not in accumulated:
                        accumulated[name] = gradient.to(primary).clone()
                    else:
                        accumulated[name].add_(gradient.to(primary))
        if not torch.equal(cpu, torch.get_rng_state()):
            raise ValueError('Pose workers changed shared CPU RNG; reject optimizer update')
        for name, parameter in self.parameters[0].items():
            parameter.grad = accumulated.get(name)
        torch.cuda.set_rng_state(initial_rng, primary)
        torch.cuda.default_generators[primary.index].set_offset(offset)
        completed = torch.cuda.get_rng_state(primary)
        for device in self.devices[1:]:
            torch.cuda.set_rng_state(completed, device)
        self.last_outputs = outputs
        return losses
