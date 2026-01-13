import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Iterator, List, Set

from cs336_systems.ddp_utils import _unique_params_by_storage, _broadcast_params

class DDPIndividualParameters(nn.Module):
    """
    DDP container that overlaps gradient communication
    """
    def __init__(self, module: torch.nn.Module, src_rank: int = 0) -> None:
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")
        
        self.module = module
        self.src_rank = src_rank
        self.world_size = dist.get_world_size()

        # Broadcast parameters and buffers from src_rank to all other ranks
        _broadcast_params(self.module, src=self.src_rank)

        # ---- Register hooks for gradient synchronization ----
        # asynchronous work handle per parameter pointer
        self._pending_works: Dict[int, dist.Work] = {}

        # track which parameters have registered hooks
        self._launched_params_ptr: Set[int] = set()

        # keep hook handles alive
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # register hooks
        self._register_grad_hooks()

    def _register_grad_hooks(self) -> None:

        params = _unique_params_by_storage(self.module.parameters())

        for p in params:
            if not p.requires_grad: # not trainable
                continue

            param_ptr = p.data.data_ptr()

            def _make_hook(param: nn.Parameter, ptr: int):
                def _grad_hook(*_):
                    
                    # if already launched for this param, skip
                    if ptr in self._launched_params_ptr:
                        return
                    
                    grad = param.grad
                    if grad is None:
                        return
                    
                    # initiate asynchronous all-reduce
                    work = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)

                    self._pending_works[ptr] = work
                    # mark as launched
                    self._launched_params_ptr.add(ptr)
                
                    return
                
                return _grad_hook

            # p.register_hook hooks the gradient computation for this parameter
            handle = p.register_post_accumulate_grad_hook(_make_hook(p, param_ptr))

            self._hook_handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self) -> None:
        """
        Wait for all gradient communications to finish and then average the gradients.
        """
        # wait for all pending works
        for ptr, work in self._pending_works.items():
            work.wait()

        ########
        # print(f"len(self._launched_params_ptr) = {len(self._launched_params_ptr)}")
        # print(f"len(list(self.module.parameters())) = {len(list(self.module.parameters()))}")
        #######
        
        for p in _unique_params_by_storage(self.module.parameters()):
            if not p.requires_grad or p.grad is None:
                continue
            param_ptr = p.data.data_ptr()
            if param_ptr in self._pending_works:
                p.grad.div_(self.world_size)

        self._pending_works.clear()
        self._launched_params_ptr.clear()

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse)
        # yield from _unique_params_by_storage(self.module.parameters())
    
    def buffers(self, recurse: bool = True):
        return self.module.buffers(recurse)
    
    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)
