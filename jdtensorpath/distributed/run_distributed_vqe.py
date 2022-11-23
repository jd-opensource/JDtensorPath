import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
import os
import torch
from jdtensorpath.distributed import rpc_contract, rpc_contract_GPU
import itertools



class Observer:
    def __init__(self):
        self.cost_func = None

    def set_circuit(self, cost_func):

        # self._requires_grad = kwargs.get("requires_grad", True)

        # which output to be used to calculate backward
        # self._backward_index = kwargs.get("backward_index", None)

        self.cost_func = cost_func.to_here()#.to_here()
        #print(type(self.cost_func), type(cost_func), self.cost_func)
        return True

    # parameters: tuple
    def execute(self, parameters):

        # world_size = int(os.environ['world_size'])
        # rank = int(os.environ['rank'])
        num_gpus = int(os.environ['num_gpus'])

        host_device = parameters[0].device
        # available_gpus = [torch.device('cuda:'+''.join(str(i))) for i in range(num_gpus)]
        available_gpus = [torch.device('cpu') for i in range(num_gpus)]
        results = []
        for i in range(num_gpus):
            gpu_params = tuple(par.to(available_gpus[i]) for par in parameters)
            tmpt = self.cost_func(*gpu_params)
            results.append(tmpt)

        return [tmpt.to(host_device) for tmpt in results]

        

class run_distributed:
    def __init__(self, world_size, rank, num_gpus=0, master_addr='localhost', master_port='8119'):#master_addr='172.17.224.178', master_port='8119'

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['world_size'] = str(world_size)
        os.environ['num_gpus'] = str(num_gpus)
        os.environ['rank'] = str(rank)
        self._world_size = world_size
        self._rank = rank
        # print("here1.1")
        # print(self._world_size, self._rank)
        if self._rank == 0:
            rpc.init_rpc("master", rank=0, world_size=self._world_size)
        else:
            name = f"worker_{rank}"
            rpc.init_rpc(name, rank=self._rank, world_size=self._world_size)
        # print("here1.2")

    def set_circuit(self, quantum_circuit, **kwargs):


        self._requires_grad = kwargs.get("requires_grad", True)

        # which output to be used to calculate backward
        self._backward_index = kwargs.get("backward_index", None)


        self.ob_rrefs = []

        for ob_rank in range(1, self._world_size): # neglect the master computer

            # put data into corresponding thread
            which_rank = ob_rank
            name = f"worker_{which_rank}"

            self.ob_rrefs.append(rpc.remote(name, Observer))



        quantum_circuit = RRef(quantum_circuit)
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                # rpc.rpc_async(
                #     ob_rref.owner(),
                #     ob_rref.rpc_sync().set_circuit,
                #     args=(quantum_circuit,)
                # )
                ob_rref.rpc_async().set_circuit(quantum_circuit)
            )

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()

    def set_cost_func(self, cost_func):
        self.cost_func = cost_func



    
    # Notice, parameters must be in CPU since pytorch distributed do not support cuda tensor copy yet!!
    def __call__(self, qauntum_parameters=None, cost_parameters=None):
        if self._rank != 0:
            raise ValueError("Only master node need to call!!")

        if qauntum_parameters == None:
            qauntum_parameters = tuple()

        if cost_parameters == None:
            cost_parameters = tuple()

        # No need to keep track of gradient information
        if self._requires_grad is False:
            # print("NO grad")

            futs = []
            for ob_rref in self.ob_rrefs:
                # make async RPC to kick off an episode on all observers
                futs.append(
                    # rpc.remote(
                        # ob_rref.owner(),
                        # ob_rref.rpc_sync().execute,
                        # args=parameters
                    # )
                    ob_rref.remote().execute(quantum_parameters)
                )


            result = torch.sum([fut.to_here() for fut in futs])

            return result

        else:
            with dist_autograd.context() as context_id:

                futs = []
                for ob_rref in self.ob_rrefs:
                    # make async RPC to kick off an episode on all observers
                    futs.append(
                        # rpc.rpc_sync(
                        #     ob_rref.owner(),
                        #     ob_rref.rpc_sync().execute,
                        #     args=parameters
                        # )
                        ob_rref.remote().execute(qauntum_parameters)
                    )

                # wait until all obervers have finished this episode
                #for fut in futs:
                #    fut.wait()

                results = [fut.to_here() for fut in futs]
                results = list(itertools.chain.from_iterable(results))
                #print(results)

                result = self.cost_func(results, *cost_parameters)

                # which output to be used to calculate backward
                if self._backward_index is not None:
                    loss = result[self._backward_index]
                else:
                    loss = result

                dist_autograd.backward(context_id, [loss])
                grads = dist_autograd.get_gradients(context_id)

            #print(grads)
            #print(parameters)
            #print(1, qauntum_parameters[0].grad)
            for param in qauntum_parameters:

                if param.requires_grad:
                    if param.grad is not None:
                        param.grad += grads[param]
                    else:
                        param.grad = grads[param]
            #print(2, qauntum_parameters[0].grad)

            return result
    
    def shutdown(self):
        rpc.shutdown()