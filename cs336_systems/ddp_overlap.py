import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Iterator, List, Set, Any
from dataclasses import dataclass

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

                    self._pending_works[ptr] = work # type: ignore
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
    


@dataclass
class _ParamSlice:
    """
    Helper dataclass to hold parameter slice information.
    """
    bucket_idx: int
    offset: int
    numel: int
    # shape: torch.Size


class DDPBucketedParameters(nn.Module):
    """
    DDP container that overlaps gradient communication in buckets.
    """
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float, src_rank: int = 0) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")
        
        self.module = module
        self.src_rank = src_rank
        self.world_size = dist.get_world_size()

        # ---- Broadcast parameters and buffers from src_rank to all other ranks ----
        _broadcast_params(self.module, src=self.src_rank)

        # ---- Build buckets ----
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        
        # param_ptr to bucket
        self._param_slices: Dict[int, _ParamSlice] = {}
        
        # buckets
        # self._buckets: List[_ParamBucket] = []
        self.buckets: List[Dict[str, Any]] = []

        # track which parameters have registered hooks
        self._launched_params_ptr: Set[int] = set()

        # map from param ptr to bucket idx and offset
        self.param_to_bucket: Dict[int, int] = {}

        # track pending works
        self._pending_works: List[dist.Work] = []
        # keep hook handles alive
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        self._build_buckets()
        self._register_grad_hooks()


    def _build_buckets(self):
        """
        build buckets for parameters
        """
        # use reversed order to match backward pass
        unique_params = _unique_params_by_storage(self.module.parameters())
        params = unique_params[::-1]

        if len(params) == 0:
            return
        
        current_bucket_params: List[nn.Parameter] = []
        current_bucket_size: int = 0

        for p in params:
            if not p.requires_grad:
                continue
            
            param_size = p.numel() * p.element_size()

            # if adding this param exceeds bucket size, create new bucket
            if current_bucket_size + param_size > self.bucket_size_bytes and current_bucket_params:
                # create bucket buffer
                bucket_info = self._create_bucket_buffer(current_bucket_params)

                # add bucket
                self.buckets.append(bucket_info)
                
                # reset current bucket
                current_bucket_params = []
                current_bucket_size = 0 
            
            # add param to current bucket
            current_bucket_params.append(p)
            current_bucket_size += param_size
            
        # add remaining params to a bucket
        if current_bucket_params:
            bucket_info = self._create_bucket_buffer(current_bucket_params)
            self.buckets.append(bucket_info)
    
    def _create_bucket_buffer(self, bucket_params: List[nn.Parameter]):
        """
        Create a contiguous buffer for the given parameters.
        """
        # assigne storage for current bucket
        bucket_idx = len(self.buckets)
        bucket_numel = sum(p.numel() for p in bucket_params)

        bucket_buffer = torch.empty((bucket_numel, ), device=bucket_params[0].device, dtype=bucket_params[0].dtype)
        bucket_info = {
            "buffer": bucket_buffer,
            "params": bucket_params,
            # "size": bucket_size,    
            "ready_count": 0,   # number of params ready in the bucket
            "work": None,    # hook work handle for the bucket
        }
        offset = 0
        for param in bucket_params:
            param_ptr = param.data.data_ptr()
            # self.param_to_bucket[param_ptr] = bucket_idx
            # map param to bucket slice
            self._param_slices[param_ptr] = _ParamSlice(bucket_idx=bucket_idx, offset=offset, numel=param.numel())
            offset += param.numel()

        return bucket_info
    
    def _register_grad_hooks(self) -> None:
        """
        register gradient hooks for parameters to handle bucketed communication
        """
        params = _unique_params_by_storage(self.module.parameters())

        for p in params:
            if not p.requires_grad: # not trainable
                continue

            param_ptr = p.data.data_ptr()

            def _make_hook(param: nn.Parameter, ptr: int):
                def _grad_hook(*_):
                    if ptr in self._launched_params_ptr:
                        return
                    # get bucket info
                    param_slice = self._param_slices[ptr]
                    bucket_info = self.buckets[param_slice.bucket_idx]

                    # copy gradient to bucket buffer
                    bucket_buffer = bucket_info["buffer"]
                    grad = param.grad
                    if grad is None:
                        return
                    offset = param_slice.offset
                    numel = param_slice.numel
                    bucket_buffer[offset:offset+numel].copy_(grad.reshape(-1))

                    # rebind param.grad to view into bucket_buffer
                    param.grad = bucket_buffer[offset:offset+numel].view_as(param.grad)

                    # mark parameter as ready
                    self._launched_params_ptr.add(ptr)
                    bucket_info["ready_count"] += 1

                    # if all parameters in the bucket are ready, launch all-reduce
                    if bucket_info["ready_count"] == len(bucket_info["params"]):
                        # initiate asynchronous all-reduce
                        work = dist.all_reduce(bucket_buffer, op=dist.ReduceOp.SUM, async_op=True)
                        bucket_info["work"] = work
                        self._pending_works.append(work)

                    return
                return _grad_hook
            
            # p.register_hook hooks the gradient computation for this parameter
            handle = p.register_post_accumulate_grad_hook(_make_hook(p, param_ptr))
            self._hook_handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self) -> None:
        """
        wait for all gradient communications to finish; average the gradients.
        """
        for work in self._pending_works:
            work.wait()

        # average gradients from buckets
        for bucket_info in self.buckets:
            if bucket_info["work"] is None:
                continue
            bucket_buffer = bucket_info["buffer"]

            # average
            bucket_buffer.div_(self.world_size)

            # # copy back to parameters
            # for param in bucket_info["params"]:
            #     offset = self._param_slices[param.data.data_ptr()].offset
            #     numel = param.numel()
            #     param.grad.data.copy_(bucket_buffer[offset: offset+numel].view_as(param.grad.data))
            
            # reset bucket info
            bucket_info["ready_count"] = 0
            bucket_info["work"] = None
        
        self._pending_works.clear()
        self._launched_params_ptr.clear()

    def __getattr__(self, name):
        if name == "module":
            return super().__getattr__(name)
        return getattr(self.__dict__["module"], name)

                    




