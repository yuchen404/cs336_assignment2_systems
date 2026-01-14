import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from typing import Iterable, Set, Tuple, List

def cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

def _unique_params_by_storage(params: Iterable[nn.Parameter]) -> List[nn.Parameter]:
    """
    Return unique parameters based on their underlying storage
    to avoid redundant all-reduce operations on tied weights.
    """
    seen_storages = set()  # type: Set[int]
    unique_params = []
    for p in params:
        ptr = p.data.data_ptr()
        if ptr not in seen_storages:
            seen_storages.add(ptr)
            unique_params.append(p)
    return unique_params

def _broadcast_params(model: nn.Module, src: int = 0) -> None:
    """
    Broadcast model parameters from src to all other processes.
    """
    # Broadcast parameters
    for param in _unique_params_by_storage(model.parameters()):
        dist.broadcast(param.data, src)

    # Broadcast buffers
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src)

def get_ddp_individual_parameters(model: nn.Module, src: int = 0) -> nn.Module:
    """
    Get parameter synchronized data-distributed model
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")
    _broadcast_params(model, src)

    return model

def ddp_individual_parameters_on_after_backward(model: nn.Module) -> None:
    """
    Gradient synchronization after backward pass for ddp
    communication call for each parameter
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")
    
    # number of processes
    world_size = dist.get_world_size()

    # cnt = 0
    # all-reduce on parameters
    for param in _unique_params_by_storage(model.parameters()):
        if not param.requires_grad:
            continue
        if param.grad is not None:
            # cnt += 1
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=False)
            param.grad.div_(world_size)
    # if dist.get_rank() == 0:
    #     print("all_reduce calls:", cnt)

    # dist.barrier()

def ddp_flattened_parameters_on_after_backward(model: nn.Module) -> None:
    """
    Gradient synchronization after backward pass for ddp
    single all-reduce call for all parameters
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")
    
    # number of processes
    world_size = dist.get_world_size()

    # flatten all gradients
    params = _unique_params_by_storage(model.parameters())
    grads = []
    for p in params:
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad)
    
    if len(grads) == 0:
        return
    
    flat_grads = _flatten_dense_tensors(grads)

    # all-reduce on flattened gradients
    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=False)
    flat_grads.div_(world_size)

    # unflatten back to individual gradients
    unflat_grads = _unflatten_dense_tensors(flat_grads, grads)
    for p, g in zip(grads, unflat_grads):
        p.copy_(g)

# def ddp_flattened_parameters_on_after_backward(model: nn.Module) -> None:
#     """
#     Flatten-grad all-reduce with caching to reduce per-step allocation overhead.

#     Cache fields stored on model:
#       - _ddp_flat_cache_grads: list[Tensor] references to grads to sync
#       - _ddp_flat_cache_numels: list[int] numel per grad
#       - _ddp_flat_buffer: flat Tensor used every step
#     """
#     if not dist.is_initialized():
#         raise RuntimeError("Distributed process group is not initialized.")

#     world_size = dist.get_world_size()

#     # ---- Build cache on first call (or if grads structure changes) ----
#     cache_ready = hasattr(model, "_ddp_flat_buffer") and hasattr(model, "_ddp_flat_cache_grads")

#     if not cache_ready:
#         params = _unique_params_by_storage(model.parameters())
#         grads: List[torch.Tensor] = []
#         numels: List[int] = []

#         for p in params:
#             if p.requires_grad and (p.grad is not None):
#                 g = p.grad
#                 if not g.is_contiguous():
#                     # make contiguous once; keep it contiguous afterwards
#                     p.grad = g.contiguous()
#                     g = p.grad
#                 grads.append(g)
#                 numels.append(g.numel())

#         if len(grads) == 0:
#             return

#         # Allocate a persistent flat buffer on the same device/dtype as grads[0]
#         total_numel = sum(numels)
#         flat_buf = torch.empty(total_numel, device=grads[0].device, dtype=grads[0].dtype)

#         model._ddp_flat_cache_grads = grads
#         model._ddp_flat_cache_numels = numels
#         model._ddp_flat_buffer = flat_buf

#     grads = model._ddp_flat_cache_grads
#     numels = model._ddp_flat_cache_numels
#     flat_buf = model._ddp_flat_buffer

#     # ---- Pack grads into flat buffer (copy) ----
#     # NOTE: still a copy, but avoids re-allocation of flat tensor each step.
#     offset = 0
#     for g, n in zip(grads, numels):
#         flat_buf[offset: offset + n].copy_(g.view(-1))
#         offset += n

#     # ---- One all-reduce ----
#     dist.all_reduce(flat_buf, op=dist.ReduceOp.SUM, async_op=False)
#     flat_buf.div_(world_size)

#     # ---- Unpack back into grads (copy) ----
#     offset = 0
#     for g, n in zip(grads, numels):
#         g.view(-1).copy_(flat_buf[offset: offset + n])
#         offset += n