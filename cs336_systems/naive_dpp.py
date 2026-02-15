import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"    # Adjust as needed
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from copy import deepcopy

from typing import Iterable, Set, Tuple, List
from cs336_systems.ddp_utils import (
    get_ddp_individual_parameters,
    ddp_individual_parameters_on_after_backward,
)


def _setup_process_group(rank: int, world_size: int, backend: str = "gloo") -> torch.device:
    """
    Single node multiprocess
    Cpu/Gloo or GPU/NCCL
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)
    else:
        return torch.device("cpu")
    
def _cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

# ---- Toy model ----
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
        )   
    def forward(self, x) -> torch.Tensor:
        return self.net(x)
    
def make_toy_data(seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    inputs_x = torch.randn(80, 20, generator=g)
    true_w = torch.randn(20, 15, generator=g)
    inputs_y = inputs_x @ true_w + torch.randn(80, 15, generator=g) * 0.1
    return inputs_x, inputs_y

def single_process_train(model: nn.Module, inputs_x: torch.Tensor, inputs_y: torch.Tensor, steps: int = 5) -> nn.Module:
    model = deepcopy(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for i in range(steps):
        optimizer.zero_grad()
        out = model(inputs_x)
        loss = loss_fn(out, inputs_y)
        loss.backward()
        optimizer.step()
        torch.manual_seed(42 + i)  # change data each step
        idx = torch.randperm(inputs_x.size(0))
        inputs_x, inputs_y = inputs_x[idx], inputs_y[idx]

    return model

def ddp_worker(rank: int, world_size: int, backend: str = "gloo") -> None:
    
    device = _setup_process_group(rank, world_size, backend)
    dist.barrier()

    torch.manual_seed(rank)

    # Initialize models
    model = ToyModel().to(device)
    steps = 5
    non_parallel_model = deepcopy(model).to(device)

    # ddp model local init, but overwrite with broadcasted params from src
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = get_ddp_individual_parameters(ddp_base, src=0)

    data_x, data_y = make_toy_data(seed=0)
    data_x = data_x.to(device)
    data_y = data_y.to(device)

    # Baseline single-process training
    if rank == 0:
        baseline_single = single_process_train(non_parallel_model, data_x, data_y, steps=steps)
    else:
        baseline_single = None

    # Shard the batch among workers
    assert data_x.size(0) % world_size == 0, "Batch size must be divisible by world size"
    local_bs = data_x.size(0) // world_size

    loss_fn = nn.MSELoss()
    ddp_opt = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training loop
    for i in range(steps):
        ddp_opt.zero_grad()

        # sharded data
        local_x = data_x[rank * local_bs: (rank + 1) * local_bs]
        local_y = data_y[rank * local_bs: (rank + 1) * local_bs]

        out = ddp_model(local_x)
        loss = loss_fn(out, local_y)
        loss.backward()

        # Gradient synchronization
        ddp_individual_parameters_on_after_backward(ddp_model)
        
        # Optimizer step
        ddp_opt.step()

        # Change data each step
        torch.manual_seed(42 + i)
        idx = torch.randperm(data_x.size(0))
        data_x, data_y = data_x[idx], data_y[idx]

    # Validate equivalence with baseline single-process model
    dist.barrier()
    if rank == 0:
        for param_single, param_dpp in zip(baseline_single.parameters(), ddp_model.parameters()):
            if param_single.requires_grad and param_dpp.requires_grad:
                # Check trainable parameters are close
                assert torch.allclose(param_single, param_dpp, atol=1e-6, rtol=1e-6), "Parameters do not match!"
            else:
                # Check non-trainable parameters are exactly equal
                assert torch.allclose(param_single, param_dpp, atol=0, rtol=0), "Non-trainable parameters do not match!"
        print(f"Rank {rank}: DDP model matches single-process model, backend={backend}")
    
    dist.barrier()
    _cleanup_process_group()


def main():
    backend = "gloo"
    world_size = 4  

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    # backend = "nccl"
    # world_size = 3  # limited by visible GPUs

    mp.spawn(ddp_worker, args=(world_size, backend), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

    