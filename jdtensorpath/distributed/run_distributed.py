import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
import os
from tempfile import NamedTemporaryFile
import torch
from jdtensorpath.distributed import rpc_contract, rpc_contract_GPU
#from .helpers import alter
import itertools
import warnings



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
    def execute(self, parameters, gpus_per_cpu):
        '''
        parameters: list of tuple of torch.tensor; or tuple of torch.tensor
        '''

        # world_size = int(os.environ['world_size'])
        # rank = int(os.environ['rank'])
        # num_gpus = int(os.environ['num_gpus'])
        #print("parameters is: ", parameters)


        # state vector mode or single GPU/CPU tensor network mode
        # use one CPU or GPU for the whole circuit calculation once
        # different data is paralleled in multiple GPUs or CPUs across multiple computers
        # this is for small quantum circuit
        # GPU parallelism
        if gpus_per_cpu == 1:
            # needs to move data from cpu to gpu;
            # if input data in cpu, TeD-Q will run in cpu
            # if input data in gpu, TeD-Q will run in gpu

            num_data = len(parameters)
            if num_data > gpus_per_cpu:
                warnings.warn("Number of GPUs is smaller than assigned number of parallel data!!")

            host_device = parameters[0][0].device
            available_gpus = [torch.device('cuda:'+''.join(str(i))) for i in range(gpus_per_cpu)]
            #available_gpus = [torch.device('cpu') for i in range(gpus_per_cpu)]
            results = []
            for i in range(num_data):
                which_gpu = i%gpus_per_cpu
                gpu_params = tuple(par.to(available_gpus[which_gpu]) for par in parameters[i])
                tmpt = self.cost_func(*gpu_params)
                # make sure each thread finish its calculation before new data assigned to it.
                results.append(tmpt.to(host_device))    

            return results

        # Slice the tensor network, and use multiple GPUs in one computer for the whole circuit calculation once
        # different data can be paralled in multiple computers
        # this is for medium quantum circuit, which needs several GPUs to contained the data for one calculation.
        elif gpus_per_cpu >= 2:
            # don't need to move data from cpu to gpus
            # moving data from cpu to gpus is done in JD_opt_tn.py GPU_parallel_sliced_contract() function
            pass


        # state vector mode or single GPU/CPU tensor network mode
        # use one CPU or GPU for the whole circuit calculation once
        # different data is paralleled in multiple GPUs or CPUs across multiple computers
        # this is for small quantum circuit
        # CPU parallelism
        elif gpus_per_cpu == 0:
            # don't need to move data from cpu to gpu
            pass


        result = self.cost_func(*parameters)
        # return result in form of list
        return [result]


 






class run_distributed:
    def __init__(self, num_nodes, rank=0, gpus_per_cpu=0, cpus_per_node=1, master_addr='localhost', master_port='8119'):#master_addr='172.17.224.178', master_port='8119'

        self._num_nodes = num_nodes
        self._rank = rank
        self._gpus_per_cpu = gpus_per_cpu
        self._cpus_per_node = cpus_per_node

        # master node is not supposed to be used for calculation.
        self._world_size = self._num_nodes * self._cpus_per_node + 1

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['num_nodes'] = str(self._num_nodes)
        os.environ['rank'] = str(self._rank)
        os.environ['gpus_per_cpu'] = str(self._gpus_per_cpu)
        os.environ['cpus_per_node'] = str(self._cpus_per_node)
        
        os.environ['world_size'] = str(self._world_size)

        # print("here1.1")
        # print(self._world_size, self._rank)

        if self._world_size < 2:
            raise ValueError("world_size must larger than or equal to 2!!")

        if self._rank == 0:
            slurm_job_file = self.generate_slurm_job_file()
            str_sbatch_cmd = "sbatch " + slurm_job_file.name
            try:
                f=os.popen(str_sbatch_cmd)
            except Exception as e:
                print(e)
                print(f"Error happen in slurm!!!, please use squeue to find out job ID and then use scancel 'job ID' to cancel that slurm job!!!")
                print(f.read())
                # rais the exception and stop the program
                raise Exception("stop the program")
            else:
                slurm_job_id = f.read().split("Submitted batch job ")[1]
                self._slurm_job_id = slurm_job_id
                print(f"slurm job id is: ", self._slurm_job_id)
                print(f"If program exit unexpectlly, please use'squeue' to find out job ID and then use 'scancel <job ID>' to cancel that slurm job!!!")



            #print(slurm_job_file.read())

            slurm_job_file.close()

            try:
                rpc.init_rpc("master", rank=0, world_size=self._world_size)
            except Exception as e:
                print(e)
                print(f"RPC initialization unsuccessfully! cancel the slurm job and quit the program!")
                # cancel the job
                str_scancel_cmd = "scancel " + self._slurm_job_id
                os.popen(str_sbatch_cmd)
                # rais the exception and stop the program
                raise Exception("stop the program")

        else:
            name = f"worker_{rank}"
            rpc.init_rpc(name, rank=self._rank, world_size=self._world_size)
        # print("here1.2")

    
    # def generate_slurm_job_file(self):
    #     slurm_job_file = "/raid/slurm-for-quantum/home/qc01/cyc/TeD-Q/tedq/distributed_worker/cycslurmjob.sh"
    #     dict_str_old_new = dict()

    #     str_num_nodes = "#SBATCH --nodes=" + str(self._num_nodes) + "\n"
    #     dict_str_old_new["#SBATCH --nodes="] = str_num_nodes

    #     str_cpus_per_node = "#SBATCH --ntasks-per-node=" + str(self._cpus_per_node) + "\n"
    #     dict_str_old_new["#SBATCH --ntasks-per-node="] = str_cpus_per_node

    #     str_gpus_per_cpu = "#SBATCH --gres=gpu:" + str(self._gpus_per_cpu) + "\n"
    #     dict_str_old_new["#SBATCH --gres=gpu:"] = str_gpus_per_cpu
    
    #     alter(slurm_job_file, dict_str_old_new)

    #     return slurm_job_file

    def generate_slurm_job_file(self):

        str_num_nodes = "#SBATCH --nodes=" + str(self._num_nodes) + "\n"
        str_cpus_per_node = "#SBATCH --ntasks-per-node=" + str(self._cpus_per_node) + "\n"
        str_gpus_per_cpu = "#SBATCH --gres=gpu:" + str(self._gpus_per_cpu) + "\n"
        str_srun = ""
        str_srun += "srun python rpc_workers.py --num_nodes " + str(self._num_nodes)
        str_srun += " --rank $rank --gpus_per_cpu " + str(self._gpus_per_cpu)
        str_srun += " --cpus_per_node " + str(self._cpus_per_node)
        str_srun += " --master_addr " + master_addr
        str_srun += " --master_port " + master_port + "\n"
        
        file_data = ""
        file_data += "#!/bin/bash\n"
        file_data += "#SBATCH -o job.%j.out\n"
        file_data += "#SBATCH --partition=p40\n"
        file_data += "#SBATCH -J myFirstJob\n"
        file_data += str_num_nodes
        file_data += str_cpus_per_node
        file_data += str_gpus_per_cpu
        file_data += "cd /raid/slurm-for-quantum/home/qc01/cyc/TeD-Q/tedq/distributed_worker/\n"
        file_data += "rank=$(($SLURM_PROCID+1))\n"
        file_data += str_srun


        fp = NamedTemporaryFile(mode='w+t', encoding="utf-8", newline='\n') # 创建一个临时文件
        fp.write(file_data)     # 向该零时文件中写入一些数据
        fp.seek(0)

        return fp



    def shutdown(self):
        rpc.shutdown()

        # make sure slurm job is finished!
        str_scancel_cmd = "scancel " + self._slurm_job_id
        os.popen(str_sbatch_cmd)




class run_distributed_circuit_parallel(run_distributed):


    def set_circuit(self, quantum_circuit):
        try:
            self._set_circuit(quantum_circuit)
        except Exception as e:
            print(e)
            print(f"set_circuit unsuccessfully! cancel the slurm job and quit the program!")
            # cancel the job
            str_scancel_cmd = "scancel " + self._slurm_job_id
            os.popen(str_sbatch_cmd)
            # rais the exception and stop the program
            raise Exception("stop the program")

    def _set_circuit(self, quantum_circuit):

        self._calculation_mode = quantum_circuit.calculation_mode

        if self._rank != 0:
            raise ValueError("Only master node need to set_circuit!!")

        # Slice the tensor network, and use multiple GPUs in one computer for the whole circuit calculation once
        # different data can be paralled in multiple computers
        # this is for medium quantum circuit, which needs several GPUs to contained the data for one calculation.
        if self._calculation_mode == 'GPU':
            if self._cpus_per_node != 1:
                raise ValueError(f"Option 'contract_parallel' chosen to be 'GPU', cpus_per_node must be default value 1")
            if self._gpus_per_cpu < 2:
                raise ValueError(f"gpus_per_cpu < 2, please change 'contract_parallel' to 'False'")

        # Slice the tensor network, and use multiple GPUs across computers for the whole circuit calculation once
        # this is for big quantum circuit, which needs lots of GPUs across computers to contained the data for one calculation.
        elif  self._calculation_mode == 'distributed_GPU':
            raise ValueError(f"Please use 'run_distributed_slicing_parallel' instead of 'run_distributed_circuit_parallel'")

        # Slice the tensor network, and use multiple CPUs across computers for the whole circuit calculation once
        # this is for big quantum circuit, which needs lots of CPUs across computers to do the calculation.
        elif  self._calculation_mode == 'distributed_CPU':
            raise ValueError(f"Please use 'run_distributed_slicing_parallel' instead of 'run_distributed_circuit_parallel'")

        # state vector mode or single GPU/CPU tensor network mode
        # use one CPU or GPU for the whole circuit calculation once
        # different data is paralleled in multiple GPUs or CPUs across multiple computers
        # this is for small quantum circuit
        else:
            # using CPU parallelism
            if self._gpus_per_cpu == 0:
                pass
            # using GPU parallelism
            elif self._gpus_per_cpu == 1:
                pass
            else:
                raise ValueError(f"gpus_per_cpu must be 0 for cpu parallelism or 1 for GPU parallelism!!")



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
                ob_rref.rpc_async().set_circuit(quantum_circuit)
            )

        # wait until all obervers have finished
        for fut in futs:
            fut.wait()

    def set_cost_func(self, cost_func, **kwargs):

        try:
            self._set_cost_func(cost_func, **kwargs)
        except Exception as e:
            print(e)
            print(f"set_cost_func unsuccessfully! cancel the slurm job and quit the program!")
            # cancel the job
            str_scancel_cmd = "scancel " + self._slurm_job_id
            os.popen(str_sbatch_cmd)
            # rais the exception and stop the program
            raise Exception("stop the program")


    def _set_cost_func(self, cost_func, **kwargs):

        if self._rank != 0:
            raise ValueError("Only master node need to set_cost_func!!")

        self._requires_grad = kwargs.get("requires_grad", True)

        # which output to be used to calculate backward
        self._backward_index = kwargs.get("backward_index", None)

        self.cost_func = cost_func



    def __call__(self, quantum_parameters, cost_parameters=None):
        try:
            self._call_func(quantum_parameters, cost_parameters)
        except Exception as e:
            print(e)
            print(f"call_func unsuccessfully! cancel the slurm job and quit the program!")
            # cancel the job
            str_scancel_cmd = "scancel " + self._slurm_job_id
            os.popen(str_sbatch_cmd)
            # rais the exception and stop the program
            raise Exception("stop the program")

    
    # Notice, parameters must be in CPU since pytorch distributed do not support cuda tensor copy yet!!
    def _call_func(self, quantum_parameters, cost_parameters=None):
        '''
        quantum_parameters: list of tuple
        cost_parameters: quantum_parameters
        '''
        if self._rank != 0:
            raise ValueError("Only master node need to call!!")

        num_data = len(quantum_parameters)

        if num_data == 0:
            raise ValueError("length of quantum_parameters must > 0, if no parameters needed, try [tuple(),tuple(),...]")


        if cost_parameters == None:
            cost_parameters = tuple()
            


        # No need to keep track of gradient information
        if self._requires_grad is False:
            # print("NO grad")

            results = self.real_execute(quantum_parameters, cost_parameters, num_data)

            return results

        else:
            with dist_autograd.context() as context_id:

                results = self.real_execute(quantum_parameters, cost_parameters, num_data)

                # which output to be used to calculate backward
                if self._backward_index is not None:
                    loss = results[self._backward_index]
                else:
                    loss = results

                dist_autograd.backward(context_id, [loss])
                grads = dist_autograd.get_gradients(context_id)
                #print("length of gradients:  ", len(grads))
                #print(grads)


            # using set to make sure each parameter only appears once;
            for param in set(itertools.chain.from_iterable(quantum_parameters)):

                if param.requires_grad:
                    if param.grad is not None:
                        param.grad += grads[param]
                    else:
                        param.grad = grads[param]

            return results

    def real_execute(self, quantum_parameters, cost_parameters, num_data):

        futs = []
        remote_futs = []



        # the observer execute() code was originally written for self._gpus_per_cpu >= 1;
        # which is one cpu on one node and it controls all gpus on that node;
        # but later the scenario changed to one cpu controls one gpus;
        # this format is for in coordinated with the execute() code
        if self._gpus_per_cpu == 1:
            total_gpus = (self._world_size-1) * self._gpus_per_cpu

            if num_data > total_gpus:
                warnings.warn("Total number of parallel data is larger than total number of GPUs")

            list_q_pars = [[] for _ in range(self._world_size-1)]
            for i in range(num_data):
                observer = i%(self._world_size-1) # count from 0
                list_q_pars[observer].append(quantum_parameters[i])

            for j in range(self._world_size-1):
                ob_rref = self.ob_rrefs[j]

                # make async RPC to kick off an episode on all observers
                futs.append(
                    ob_rref.remote().execute(list_q_pars[j], self._gpus_per_cpu)
                )

        # 0 for CPU parallelism, cpus_per_node must be 1;
        # 1 for GPU parallelism, cpus_per_node must not larger than num of gpus in each node.
        else:

            total_size = self._world_size-1
            if num_data > total_size:
                warnings.warn("Total number of parallel data is larger than total number of assigned CPUs")

            for i in range(num_data):
                observer = i%(self._world_size-1) # count from 0
                ob_rref = self.ob_rrefs[observer]

                # make async RPC to kick off an episode on all observers
                remote_futs.append(
                    ob_rref.remote().execute(quantum_parameters[i], self._gpus_per_cpu)
                )

                # make a sychronization so that the data in the same thread will not be conflicted.
                # Each thread must finish its calculation before new data assigned to it.
                if observer == self._world_size-2:
                    futs.extend([fut.to_here() for fut in remote_futs])
                    remote_futs = []

            futs.extend([fut.to_here() for fut in remote_futs])





        quantum_results = futs
        quantum_results = list(itertools.chain.from_iterable(quantum_results))

        results = self.cost_func(quantum_results, *cost_parameters)

        return results








class run_distributed_slicing_parallel(run_distributed):

    def set_cost_func(self, cost_func, **kwargs):

        try:
            self._set_cost_func(cost_func, **kwargs)
        except Exception as e:
            print(e)
            print(f"set_cost_func unsuccessfully! cancel the slurm job and quit the program!")
            # cancel the job
            str_scancel_cmd = "scancel " + self._slurm_job_id
            os.popen(str_sbatch_cmd)
            # rais the exception and stop the program
            raise Exception("stop the program")


    def _set_cost_func(self, cost_func, **kwargs):

        if self._rank != 0:
            raise ValueError("Only master node need to set_circuit!!")

        self._requires_grad = kwargs.get("requires_grad", True)

        # which output to be used to calculate backward
        self._backward_index = kwargs.get("backward_index", None)

        self.cost_func = cost_func

        # Slice the tensor network, and use multiple GPUs across computers for the whole circuit calculation once
        # this is for big quantum circuit, which needs lots of GPUs across computers to contained the data for one calculation.
        # 'distributed_GPU':
        if self._gpus_per_cpu > 0:
            if self._cpus_per_node != 1:
                raise ValueError("Option 'contract_parallel' chosen to be 'distributed_GPU', cpus_per_node must be default value 1")
        

        # Slice the tensor network, and use multiple CPUs across computers for the whole circuit calculation once
        # this is for big quantum circuit, which needs lots of CPUs across computers to do the calculation.
        # 'distributed_CPU'
        if self._gpus_per_cpu == 0:
            pass


    def __call__(self, *parameters):
        try:
            self._call_func(*parameters)
        except Exception as e:
            print(e)
            print(f"call_func unsuccessfully! cancel the slurm job and quit the program!")
            # cancel the job
            str_scancel_cmd = "scancel " + self._slurm_job_id
            os.popen(str_sbatch_cmd)
            # rais the exception and stop the program
            raise Exception("stop the program")
    
    # Notice, parameters must be in CPU since pytorch distributed do not support cuda tensor copy yet!!
    def _call_func(self, *parameters):
        if self._rank != 0:
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
