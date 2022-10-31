import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
import os
import torch
from jdtensorpath.distributed import rpc_contract, rpc_contract_GPU




class run_distributed:
    def __init__(self, world_size, rank, num_gpus=0, master_addr='172.17.224.178', master_port='8119'):#master_addr='172.17.224.178', master_port='8119'

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

    def set_circuit(self, cost_func, **kwargs):

        if self.rank != 0:
            raise ValueError("Only master node need to set_circuit!!")

        self._requires_grad = kwargs.get("requires_grad", True)

        # which output to be used to calculate backward
        self._backward_index = kwargs.get("backward_index", None)

        self.cost_func = cost_func


    
    # Notice, parameters must be in CPU since pytorch distributed do not support cuda tensor copy yet!!
    def __call__(self, *parameters):
        if self.rank != 0:
            raise ValueError("Only master node need to call!!")
        #print(parameters)


        # No need to keep track of gradient information
        if self._requires_grad is False:
            # print("NO grad")
            result = self.cost_func(*parameters)

            return result

        else:
            with dist_autograd.context() as context_id:
                result = self.cost_func(*parameters)
                #print(result)
                #print(self._backward_index)

                # which output to be used to calculate backward
                if self._backward_index is not None:
                    loss = result[self._backward_index]
                else:
                    loss = result

                dist_autograd.backward(context_id, [loss])
                grads = dist_autograd.get_gradients(context_id)

            #print(grads)
            #print(parameters)
            for param in parameters:

                if param.requires_grad:
                    if param.grad is not None:
                        param.grad += grads[param]
                    else:
                        param.grad = grads[param]

            return result
    
    def shutdown(self):
        rpc.shutdown()