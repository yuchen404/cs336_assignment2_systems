import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from typing import Iterable, Set, Tuple, List

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
    """Gradient synchronization after backward pass for ddp"""
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")
    
    # number of processes
    world_size = dist.get_world_size()

    # all-reduce on parameters
    for param in _unique_params_by_storage(model.parameters()):
        if not param.requires_grad:
            continue
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=False)
            param.grad.div_(world_size)

    dist.barrier()