import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from typing import Iterable, Set, Tuple, List


# --- Distributed Setup and Teardown ---
def setup_process_group(rank: int, world_size: int, backend: str, master_addr: str, master_port: str) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def teardown_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
        
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
