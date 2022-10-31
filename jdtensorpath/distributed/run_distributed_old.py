import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
import os
import torch
from jdtensorpath.distributed import rpc_contract, rpc_contract_GPU




class run_distributed:
    def __init__(self, world_size, rank, num_gpus=0, master_addr='localhost', master_port='5678'):#master_addr='172.17.224.178', master_port='8119'

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['world_size'] = str(world_size)
        os.environ['num_gpus'] = str(num_gpus)
        os.environ['rank'] = str(rank)
        self.rank = rank
        if rank == 0:
            rpc.init_rpc("master", rank=0, world_size=world_size)
        else:
            name = f"worker_{rank}"
            rpc.init_rpc(name, rank=rank, world_size=world_size)

    def set_circuit(self, cost_func, params, optimizer=optim.Adam, lr=0.1):
        if self.rank != 0:
            raise ValueError("Only master node need to set_circuit!!")

        self.cost_func = cost_func

        self._device = params[0].device
        cpu_device = torch.device("cpu")
        self.params = [p.to(cpu_device) for p in params]
        self.rref_params = [RRef(p) for p in self.params]
        self.dist_optim = DistributedOptimizer(optimizer, self.rref_params, lr=lr)

    
    def __call__(self):
        if self.rank != 0:
            raise ValueError("Only master node need to call!!")

        with dist_autograd.context() as context_id:
            loss = self.cost_func(self.rref_params[0].to_here())
            dist_autograd.backward(context_id, [loss])
            self.dist_optim.step(context_id)
        loss = loss.to(self._device)
        return loss
    
    def shutdown(self):
        rpc.shutdown()