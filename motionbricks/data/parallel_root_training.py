"""Ordered Root microbatches with a checked logical CUDA RNG stream."""

from concurrent.futures import ThreadPoolExecutor

import torch


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


class ParallelRootMicrobatches:
    def __init__(self, stages):
        if len(stages) not in (2, 4) or any(stage.stage != 'root' for stage in stages):
            raise ValueError('This executor requires two or four Root replicas')
        self.stages = stages
        self.controllers = [stage.controller for stage in stages]
        self.parameters = [{name: parameter for name, parameter in controller.named_parameters()
            if parameter.requires_grad} for controller in self.controllers]
        if any(self.parameters[0].keys() != parameters.keys() for parameters in self.parameters[1:]):
            raise ValueError('Replica trainable scopes differ')
        self.devices = [next(iter(parameters.values())).device for parameters in self.parameters]
        if len(set(self.devices)) != len(self.devices) or any(device.type != 'cuda' for device in self.devices):
            raise ValueError('Distinct CUDA devices are required')
        self.pool = ThreadPoolExecutor(max_workers=len(stages))
        self.rng_increment = None
        self.calibration_microbatches = 0

    def close(self):
        self.pool.shutdown(wait=True)

    def _calibrate(self, microbatch, seed):
        device = self.devices[0]
        cpu = torch.get_rng_state()
        before = torch.cuda.get_rng_state(device)
        generator = torch.cuda.default_generators[device.index]
        first = generator.get_offset()
        self.controllers[0].zero_grad(set_to_none=True)
        loss = self.stages[0](microbatch, seed)
        (loss*.25).backward()
        torch.cuda.synchronize(device)
        increment = generator.get_offset()-first
        unchanged_cpu = torch.equal(cpu, torch.get_rng_state())
        self.controllers[0].zero_grad(set_to_none=True)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(before, device)
        if increment <= 0 or increment % 4 or not unchanged_cpu:
            raise ValueError('Root RNG does not follow a fixed CUDA-only Philox schedule')
        self.rng_increment = increment
        self.calibration_microbatches += 1

    def backward(self, microbatches, seeds):
        if len(microbatches) != 4 or len(seeds) != 4:
            raise ValueError('Exactly four original microbatches are required')
        if any(len(batch['world']) != 64 for batch in microbatches):
            raise ValueError('The calibrated microbatch must remain64')
        primary = self.devices[0]
        if self.rng_increment is None:
            self._calibrate(microbatches[0], seeds[0])
        cpu_state = torch.get_rng_state()
        initial_rng = torch.cuda.get_rng_state(primary)
        initial_offset = torch.cuda.default_generators[primary.index].get_offset()
        workers = len(self.devices)
        for index in range(1, workers):
            with torch.no_grad(), torch.cuda.device(self.devices[index]):
                for name, parameter in self.parameters[index].items():
                    parameter.copy_(self.parameters[0][name], non_blocking=True)
        batches = [to_device(batch, self.devices[i%workers]) for i, batch in enumerate(microbatches)]
        for device in self.devices:
            torch.cuda.synchronize(device)

        def execute(index):
            worker, device = index%workers, self.devices[index%workers]
            with torch.cuda.device(device):
                self.controllers[worker].zero_grad(set_to_none=True)
                torch.cuda.set_rng_state(initial_rng, device)
                generator = torch.cuda.default_generators[device.index]
                generator.set_offset(initial_offset+index*self.rng_increment)
                loss = self.stages[worker](batches[index], seeds[index])
                (loss*.25).backward()
                torch.cuda.synchronize(device)
                if generator.get_offset() != initial_offset+(index+1)*self.rng_increment:
                    raise ValueError('CUDA RNG increment changed; do not apply an optimizer update')
                return float(loss.detach()), {name: parameter.grad.detach() if parameter.grad is not None else None
                    for name, parameter in self.parameters[worker].items()}

        accumulated, losses = {}, []
        for first in range(0, 4, workers):
            futures = [self.pool.submit(execute, first+i) for i in range(workers)]
            for future in futures:
                loss, gradients = future.result()
                losses.append(loss)
                for name, gradient in gradients.items():
                    if gradient is None:
                        continue
                    if name not in accumulated:
                        accumulated[name] = gradient.to(primary).clone()
                    else:
                        accumulated[name].add_(gradient.to(primary))
        if not torch.equal(cpu_state, torch.get_rng_state()):
            raise ValueError('Root unexpectedly consumed the shared CPU RNG')
        for name, parameter in self.parameters[0].items():
            parameter.grad = accumulated.get(name)
        torch.cuda.set_rng_state(initial_rng, primary)
        torch.cuda.default_generators[primary.index].set_offset(initial_offset+4*self.rng_increment)
        if workers == 4:
            # Checkpoints expose one logical stream on every replica. Future
            # microbatch starts are still independently derived from the master.
            completed_rng = torch.cuda.get_rng_state(primary)
            for device in self.devices[1:]:
                torch.cuda.set_rng_state(completed_rng, device)
        return losses
