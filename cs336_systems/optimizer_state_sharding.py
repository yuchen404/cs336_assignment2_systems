
import torch
from torch import nn
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Type, Any, List, Optional

from cs336_systems.ddp_utils import cuda_sync_if_needed, _unique_params_by_storage

class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        """
        Initializes the sharded state optimizer. params is a collection of parameters to be optimized (or parameter
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = kwargs

        # optimizer instance for local parameters
        self._local_optimizer: Optional[Optimizer] = None

        # track parameter ownership for synchronization, ptr to rank
        self._parms_owner: dict[int, int] = {}
        # track global parameters idx as flag to shard among ranks
        self._global_param_idx = 0

        # local parameter groups for this rank
        self._local_groups = []

        defaults = dict(kwargs)

        super().__init__(params, defaults)    # call super-class constructor, which will call add_param_group

        if self._local_optimizer is None:
            self._local_optimizer = self._optimizer_cls(self._local_groups, **self._optimizer_kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        """
        This method should add a parameter group to the sharded optimizer. 
        called during construction of the sharded optimizer by the super-class constructor and called during training
        """
        super().add_param_group(param_group)
        
        raw_params = param_group['params']
        params = _unique_params_by_storage(raw_params)

        # --- Shard parameters among ranks ---
        local_group = {'params': []}
        for p in params:
            owner_rank = self._get_param_owner(p)
            if owner_rank == self.rank:
                local_group['params'].append(p)

        # Copy other hyperparameters
        for k, v in param_group.items():
            if k != 'params':
                local_group[k] = v

        if len(local_group['params']) == 0:
            return
        
        # Add to local groups
        if self._local_optimizer is None:
            self._local_groups.append(local_group)
        else:
            self._local_optimizer.add_param_group(local_group)

    def _get_param_owner(self, param: nn.Parameter) -> int:
        """
        get owner rank for a parameter based on its global index
        """
        ptr = param.data.data_ptr()
        if ptr not in self._parms_owner:
            owner_rank = self._global_param_idx % self.world_size
            self._parms_owner[ptr] = owner_rank
            self._global_param_idx += 1
        
        return self._parms_owner[ptr]


    def step(self, closure=None, **kwargs):
        """
        Calls the wrapped optimizer's step() method with the provided closure and keyword arguments. After updating the parameters, synchronize with the other ranks.
        """
        if self._local_optimizer is None:
            raise RuntimeError("Local optimizer is not innitialized.")
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        local_res = self._local_optimizer.step(**kwargs)

        # Synchronize updated parameters across all ranks
        self._sync_updated_parameters()

        return loss if loss is not None else local_res
    
    def _sync_updated_parameters(self):
        """
        Broadcast updated parameters from each rank to all other ranks.
        """
        flat_params = []
        for group in self.param_groups:
            flat_params.extend(group['params'])
        all_params = _unique_params_by_storage(flat_params)
        

        pending_works = []  # track async ops
        for p in all_params:
            owner_rank = self._get_param_owner(p)
            work = dist.broadcast(p.data, src=owner_rank, async_op=True)
            pending_works.append(work)

        # Wait for all broadcasts to complete
        for work in pending_works:
            work.wait()


