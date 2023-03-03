#   Copyright 2021-2024 Jingdong Digits Technology Holding Co.,Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# pylint: disable=invalid-name

r'''
Jingdong Optimizer for Tensor Network(JDOptTN) is a module to find the optimal contraction order for large tensor network based on hyper graph partition algorithm.

**Key features:**

* An explicit binary contraction tree object
* Dynamic slicing for parrallelism and saving memory usage

**Example**

    Please refer to the TODO for the detail description.

'''



# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods

#from functools import reduce
from copy import deepcopy
import math
import functools
import warnings

from jdtensorpath.ray_parallel import read_out_ray
from jdtensorpath.ray_parallel import get_ray

from .gen_trials import generate_trials_parallel
from .gen_trials import generate_trials
from .gen_trials import generate_slicing_trials
from .gen_trials import generate_slicing_trials_parallel
from .helpers import dec_to_bin, compute_size_by_dict, to_dynamic_base


from torch.distributed.rpc import RRef

#RAY = get_ray()

_EINSUM_SYMBOLS_BASE = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

class JDOptTN:#JD-CoTenGra
    r"""
    This is an improved version of contraction path optimizer made by tedq team. This module provide similar performance to the CoTenGra module.

    Args:
        input(int): Number of search attempt
        input(list): Input indice of the tensor network     
        size_dict(list): Shape of the tensors 
        output(list): Output indice after the tensor contraction
        precut_edges(list): TODO
        imbalance(float):Imbalance of the cut. Please refer to KaHyPar for the detailed information.
        parent_node(list):TODO
        slicing_opts: TODO
        kwargs(dict): List of optional keyword arguments.

    """
    # pylint: disable=too-many-arguments, too-many-statements, too-many-locals, too-many-branches, too-many-branches
    def __init__(self, inputs, size_dict, output = None, precut_edges = None, imbalance=0.01, parent_node = None, max_repeats = 128, search_parallel = True, slicing_opts = None, **kwargs):
        #print(output)
                                #search how many times
                        # list[list[char]]: input indices
                                            #shape of tensor
                                                       #list[list[char]]: left indice
                                                               # list[]: edges has been cut
                                                                             # (max(# of node)-avg(# of node))/avg(# of node)
                                                                                        # the parent node if 

        if output is None:
            output = []
        if precut_edges is None:
            precut_edges = []

        self.scr_arrays = inputs
        self.scr_output = output

        self._size_dict = size_dict
        self.sliced = False
        self.contract_parallel = False
        if slicing_opts:  #dict
            self.sliced = True
            repeats = slicing_opts.get('repeats', 10)
            target_size = slicing_opts.get('target_size', 2**27)
            # to reduce overflow of memory
            target_num_slices = slicing_opts.get('target_num_slices', None)
            # for parallel computing
            self.contract_parallel = slicing_opts.get('contract_parallel', False)


        #print(kwargs)
        self.search_parallel = search_parallel
        #print(self.search_parallel)
        if self.search_parallel:
            g_trials = generate_trials_parallel(self.search_parallel, max_repeats, inputs, self._size_dict, output, precut_edges, imbalance, parent_node, **kwargs)
        else:
            #print("start search trials without parallel")
            g_trials = generate_trials(max_repeats, inputs, self._size_dict, output, precut_edges, imbalance, parent_node, **kwargs)
            #print("end search trials without parallel")
        #for trial in trials:
        #    print(trial)
        

        best_cost_tree = math.inf  #large enough number (number of indice)
        best_cost_sliced = math.inf

        # make sure all the process has been finished! other KeyError 'CPU' will raise on ray
        #trials = []
        for trial in g_trials:
            #trials.append(trial)

            if self.sliced:
                #self.binary_tree = trial['binary_tree']
                #self.size = trial['size']  # this will be used in self.slice() function
                # not inplace, will NOT change contraction tree finally.
                cost = self.slice(binary_tree=trial['binary_tree'], size=trial['size'], target_size=target_size, target_num_slices=target_num_slices, repeats=128)
                #print(best_cost)
                if cost < best_cost_sliced:
                    self.best_trial = trial
                    best_cost_sliced = cost

            else:
                # cost mainly determined by size
                cost = math.log2(trial['size']+1.0e-10) + 0.1 * math.log10(trial['flops']+1.0e-10)
                if cost < best_cost_tree:
                    self.best_trial = trial
                    best_cost_tree = cost

        self.size = self.best_trial['size']
        self.flops = self.best_trial['flops']
        self.binary_tree = self.best_trial['binary_tree']


        print("log2(size) before slicing: ", math.log2(self.size+1.0e-10))
        print("log10(flops) before removed:   ", math.log10(self.flops+1.0e-10))


        self.contraction_list = []
        self.current_leaves = self.best_trial['current_leaves']
        #self.size = self.binary_tree.get_max_size()#this will be updated if sliced
        #print(self.size)
        self.num_slices = None
        self.removed = None
        #print(self.scr_output)
        
        if self.sliced:
            # inplace, will change contraction tree finally.
            self.slice_(binary_tree=self.binary_tree, size=self.size, target_size=target_size, target_num_slices=target_num_slices, repeats=repeats)

            if self.num_slices:
                print("removed indices:  ", self.removed)
                print("log2(num_slices):  ", math.log2(self.num_slices+1.0e-10))
                print("log2(size) after removed:   ", math.log2(self.size+1.0e-10))
                print("log10(flops) after removed:   ", math.log10(self.flops+1.0e-10))
        
        self.get_contraction_order()


        if self.search_parallel:
            import ray  # pylint: disable=import-outside-toplevel
            if not ray.is_initialized():
                raise ValueError("ray should be initialized before!!!")
            ray.shutdown()
            if ray.is_initialized():
                raise ValueError("ray should be shutdown!!")

            # clean the cache, so that the shutdown ray will not be used
            get_ray.cache_clear()



    

    def contract(self, arrays, backend = 'jax'):
        r"""
        To compute the contraction among the optimal order. 

        Args:
            arrays(array): elements in the tensors to be contracted.
            backend(String): Computation backend: jax or pytorch
        """
        #ids_list = []
        #id = core_contract_ray.remote(self.contraction_list, arrays, backend = backend)
        #ids_list.append(id)
        #print("here")
        #return read_out_ray(ids_list)

        #print("num_slices:  ", self.num_slices)
        if self.num_slices:

            if self.contract_parallel == 'GPU':
                if backend != 'torch':
                    raise ValueError("contract parallel only support pytorch backend!!!")

                # using multiple GPUs for parallelism, like pytorch model parallel method
                results = self.GPU_parallel_sliced_contract(arrays, backend=backend)



                # This is for using RAY framework for parallelism
                # This one only can use for forward operation, can not use backward function
                #results = self.ray_parallel_sliced_contract(arrays, backend=backend)


            elif self.contract_parallel == 'distributed_GPU':
                if backend != 'torch':
                    raise ValueError("contract parallel only support pytorch backend!!!")

                results = self.RPC_parallel_sliced_contract_GPU(arrays)

            elif self.contract_parallel == 'distributed_CPU':
                if backend != 'torch':
                    raise ValueError("contract parallel only support pytorch backend!!!")
                    
                results = self.RPC_parallel_sliced_contract_CPU(arrays)


            else:
                #print("should be here")
                results = self.single_thread_sliced_contract(arrays, backend=backend)


        else:
            #print("here ?")
            results = core_contract(self.contraction_list, arrays, backend=backend)

        #print(results)
        return results

    def get_sliced_arrays(self, arrays, i):
        r"""
        Get the sliced array from the original array.

        Args:
            arrays(array): original array
            i(int): index of the slice

        **Example**

        .. code-block:: python3

            # suppose size_dict are all 2, and scr_arrays = [['a', 'b'], ['a', 'c']], removed = ['a']
            import torch
            a = torch.tensor([[1., 2.], [3., 4.]])
            b = torch.tensor([[5., 6.], [7., 8.]])
            arrays = [a, b]

            >>> arrays
            [tensor([[1., 2.],
                    [3., 4.]]),
             tensor([[5., 6.],
                    [7., 8.]])]

            >>> get_sliced_arrays(arrays, 0)
            [tensor([[1., 2.]]),
             tensor([[5., 6.]])]

            >>> get_sliced_arrays(arrays, 1)            
            [tensor([3., 4.]]),
             tensor([7., 8.]])]

        """
        new_arrays = list(arrays)
        #new_arrays = deepcopy(arrays)

        # This only works for all the size_dict is 2.
        #loc_dict = dict(zip(self.removed, dec_to_bin(i, self.num_slices)))

        base = tuple(self._size_dict[i] for i in self.removed)
        loc_dict = dict(zip(self.removed, to_dynamic_base(i, base)))

        for k in range(len(self.scr_arrays)):
            element_selector = tuple(loc_dict.get(ix, slice(None)) for ix in self.scr_arrays[k])
            new_arrays[k] = new_arrays[k][element_selector]
            #print(new_arrays[k].shape)
        return new_arrays

    # pylint: disable=pointless-string-statement
    '''
    def core_contract_ray(self):
        return RAY.remote(self.core_contract)

    def core_contract(self, arrays, backend = 'jax'):
        r"""
        Contract the result without slice.

        Args:
            arrays(array): elements in the tensors to be contracted.
            backend(String): Computation backend: jax or pytorch
        """
        if backend == 'jax':
            import jax
            from jax import numpy as jnp

            for contraction in self.contraction_list[:-1]:#last one is transpose for final result correction
                order_operand, do_einsum, einsum_str_or_axes = contraction
                temp_operands = [arrays.pop(x) for x in order_operand]

                if do_einsum:
                    #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape)
                    #print(temp_operands)
                    result = jnp.einsum(einsum_str_or_axes[0], *temp_operands)
                    #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape, result.shape)
                else:
                    #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape)
                    #print(temp_operands)
                    result = jnp.tensordot(*temp_operands, axes=einsum_str_or_axes[0])

                #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape, result.shape)
                arrays.append(result)

            final_transpose = self.contraction_list[-1]
            result = jnp.transpose(arrays[0], final_transpose)
            return result



        if backend == 'torch':
            import torch
            from torch import tensor

            for contraction in self.contraction_list[:-1]:#last one is transpose for final result correction
                order_operand, do_einsum, einsum_str_or_axes = contraction
                temp_operands = [arrays.pop(x) for x in order_operand]

                if do_einsum:
                    #print(einsum_str_or_axes)
                    #print(temp_operands)
                    result = torch.einsum(einsum_str_or_axes[0], *temp_operands)
                else:
                    #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape)
                    #print(temp_operands)
                    result = torch.tensordot(*temp_operands, dims=einsum_str_or_axes[0])

                arrays.append(result)

            final_transpose = self.contraction_list[-1]
            result = arrays[0].permute(final_transpose)
            return result
    '''


    # pylint: disable=too-many-arguments, too-many-statements, too-many-locals, too-many-branches, too-many-branches
    def get_contraction_order(self):
        r"""
        Get the optimal contraction order.
        """
        new_scr_arrays = self.get_new_scr_arrays()
        #print(new_scr_arrays)
        #print(self.scr_output)
        for node in self.current_leaves:
            parent_node = node.parent
            if parent_node:
                parent_node.value += 1
                if parent_node.value == 2:
                    do_einsum = False
                    left = parent_node.left.outer_indices
                    right = parent_node.right.outer_indices
                    order_input = []
                    order_input.extend(left)
                    order_input.extend(right)
                    result_set = parent_node.outer_indices_set#prevent double count the same idx

                    #print(left, right, result_set)
                    #print(new_scr_arrays)

                    r_set = deepcopy(result_set)
                    #print("r_set  ",r_set)
                    result = []
                    for idx in order_input:
                        if idx in r_set:
                            r_set -= set(idx)#caution! r_set should be zero finally
                            result.append(idx)
                    #print(result)
                    if r_set:
                        print("Error!!! r_set should be None now!")

                    left_set = set(left)
                    right_set = set(right)
                    intersect = left_set & right_set

                    if not intersect or intersect & result_set:
                        do_einsum = True

                        down_dict = {}
                        for i, scr in enumerate(left_set|right_set|result_set):
                            down_dict[scr] = i

                        einsum_str = ''.join(tuple(_EINSUM_SYMBOLS_BASE[down_dict[scr]] for scr in left))+','
                        einsum_str += ''.join(tuple(_EINSUM_SYMBOLS_BASE[down_dict[scr]] for scr in right))+'->'
                        einsum_str += ''.join(tuple(_EINSUM_SYMBOLS_BASE[down_dict[scr]] for scr in result))
                        einsum_str_or_axes = (einsum_str, )

                    else:
                        do_einsum = False
                        left_pos, right_pos = [], []
                        for rm_idx in intersect:
                            left_pos.append(left.index(rm_idx))
                            right_pos.append(right.index(rm_idx))
                        axes = (tuple(left_pos), tuple(right_pos))
                        #print(right)
                        #print(axes)
                        einsum_str_or_axes = (axes, )

                    parent_node.outer_indices = result

                    # ATTENTION!!! position index2 must be found after tensor_1 has been pop!!
                    # Since pop(index1) will delete tensor_1 and then affect position of index2!!!
                    index1 = new_scr_arrays.index(left)
                    new_scr_arrays.pop(index1)
                    index2 = new_scr_arrays.index(right)
                    new_scr_arrays.pop(index2)

                    order_operand = (index1, index2)
                    new_scr_arrays.append(result)
                    #print(left,right,result)
                    self.current_leaves.append(node.parent)

                    contraction = (order_operand, do_einsum, einsum_str_or_axes)
                    self.contraction_list.append(contraction)

        #print(self.scr_output)
        #print(result)
        if set(self.scr_output) != set(result):
            #print(self.scr_output)
            #print(result)
            raise ValueError("Error!!! final result indices should be the same as user-specified output")
        transpose = tuple(map(result.index, self.scr_output))
        self.contraction_list.append(transpose)
        #print(transpose)

    @property
    def order_path(self):
        r'''
        return the contraction order path
        '''
        path = []
        for contraction in self.contraction_list[:-1]:#last one is transpose for final result correction
            order_operand, _, _ = contraction
            path.append(order_operand)
        return path

    @property
    def final_transpose(self):
        r'''
        return the transpose operation for the final output
        '''
        return self.contraction_list[-1]


    def get_new_scr_arrays(self):
        r'''
        return the new arrays of indice subscripts
        '''
        new_scr_arrays = deepcopy(self.scr_arrays)
        if self.num_slices:
            for i, _ in enumerate(new_scr_arrays):
                new_scr_arrays[i] = [index for index in new_scr_arrays[i] if index not in self.removed]        
        return new_scr_arrays




 



    def slice(self, binary_tree, size, target_size = 27, target_num_slices = None, repeats = 10, inplace = False):
        r"""
        Slice the tensor network.

        Args:
            target_size(int): TBD
            target_num_slices(int): TBD
            repeats(int): TBD
        """
        #print(target_size, target_num_slices)
        #print(self.size)
        if target_num_slices:
            target_size = min(target_size, size/target_num_slices)
            #print(math.log2(size), math.log2(target_size))

        if size > target_size:  # pylint: disable=no-else-return

            sliced_sets = []    

            all_nodes = binary_tree.get_all_nodes()
            for node in all_nodes:
                node_size = compute_size_by_dict(node.outer_indices_set, self._size_dict)
                if node_size > target_size:
                    sliced_set = deepcopy(node.outer_indices_set)
                    sliced_sets.append(sliced_set)  

            # for slice of each trial, inplace is False, don't need to do parallel
            # because CPU is occupied by other kahypar trials
            # if using parallel, then KeyError 'CPU' will occur on ray
            if self.search_parallel and inplace:
                slice_trials = generate_slicing_trials_parallel(self.search_parallel, repeats, sliced_sets, self.scr_output, target_size, self._size_dict)
            else:
                slice_trials = generate_slicing_trials(repeats, sliced_sets, self.scr_output, target_size, self._size_dict)

            best_cost = math.inf  # large enough number
            best_trial = None
            for trial in slice_trials:
                num_slices = trial['num_slices']
                removed = list(trial['removed'])
                # calculate the flops when slice the removed indices
                flops = binary_tree.get_flops(size_dict=self._size_dict, removed=removed)
                cost = num_slices * flops #
                if cost < best_cost:
                    best_cost = cost
                    best_trial = trial

            if inplace:
                self.removed = list(best_trial['removed'])
                # modify the contraction tree
                self.binary_tree.removed_indices(self.removed)
                self.size = best_trial['size']
                self.num_slices = best_trial['num_slices']
                s_flops = self.binary_tree.get_flops(self._size_dict)
                self.flops = self.num_slices * s_flops
                #print(self.flops, best_cost)

            return best_cost

        else:
            flops = binary_tree.get_flops(self._size_dict)
            return flops

    slice_ = functools.partialmethod(slice, inplace=True)




    def ray_parallel_sliced_contract(self, arrays, backend = 'torch'):
        r'''
        using ray framwork to do parallel contraction for sliced arrays.
        '''

        ray = get_ray(self.search_parallel)
        num_cpu = int(ray.available_resources()['CPU'])
        core_contract_ray = ray.remote(core_contract)
        ids_list = []
        result_list = []
        for i in range(self.num_slices):
            new_arrays = self.get_sliced_arrays(arrays, i)
            ids_list.append(core_contract_ray.remote(self.contraction_list, new_arrays, backend = backend))
            if len(ids_list) > num_cpu + 3:
                tmpt = read_out_ray(ids_list)
                result_list.append(tmpt)                    
        while len(ids_list) > 0:
            tmpt = read_out_ray(ids_list)
            #print(tmpt)
            result_list.append(tmpt)
        result = result_list[0].detach()
        for res in result_list[1:]:
            result += res
        return result

    def RPC_parallel_sliced_contract_CPU(self, arrays):
        #print("using RPC_parallel_sliced_contract_CPU")
        from .distributed import rpc_contract
        import torch.distributed.rpc as rpc
        import os
        world_size = int(os.environ['world_size'])
        rank = int(os.environ['rank'])
        if world_size < 2:
            raise ValueError("world_size must larger than 2!!")
        if rank != 0:
            raise ValueError("Master must be in rank 0!!")

        total_size = world_size - 1

        if self.num_slices > total_size:
            warnings.warn("Total number of slices is larger than total number of assigned CPUs")

        
        # print(' ')
        # print(' ')
        # s_arrs = []
        # r_arrs = []
        # for i in range(self.num_slices):
        #     sliced_arrays = self.get_sliced_arrays(arrays, i)
        #     s_arrs.append(sliced_arrays)
        #     #r_arrays = RRef(sliced_arrays)
        #     r_arrays = [RRef(arr) for arr in sliced_arrays]
        #     r_arrays = tuple(arrays)
        #     r_arrs.append(r_arrays)

        results = []
        remote_results = []
        for i in range(self.num_slices):

            # get the corresponding sliced arrays
            sliced_arrays = self.get_sliced_arrays(arrays, i)
            #new_arrays = list(sliced_arrays)

            # put data into corresponding thread
            # neglect the master computer  
            which_rank = i%(world_size-1) + 1
            name = f"worker_{which_rank}"
            #print(name)
            #print(self.contraction_list[:10])

            #name = f"master"

            #r_arrays = RRef(new_arrays)

            # do the contraction
            rref1 = rpc.remote(name, rpc_contract, args=(self.contraction_list, sliced_arrays))
            #print(type(rref1))
            #tmpt = rref1.to_here()
            #print(type(tmpt))
            #print(" ")
            #print(tmpt)
            remote_results.append(rref1)

            # make a sychronization so that the data in the same thread will not be conflicted.
            # Each thread must finish its calculation before new data assigned to it.
            if which_rank == world_size-1:
                results.extend([r.to_here() for r in remote_results])
                remote_results = [] # clean it since the results have been transfer back.

            # l3 = [a - b for a, b in zip(tmpt, sliced_arrays)]
            # l4 = [c**2 for c in l3]
            # l5 = [d.any() for d in l4]
            # print(any(l5))
            # for j in range(len(l5)):
            #     if l5[j]:
            #         print(j)
            #         print(len(arrays))
            #         print(tmpt[j])
            #         #print(s_arrs[i][j])
            #         #print(r_arrs[i][j])
            #         print(sliced_arrays[j])
            #         print(" ")

            #test = rpc_contract(self.contraction_list, sliced_arrays)
            #print(test)
            #print(" ")

            #l6 = [a - b for a, b in zip(test, new_arrays)]
            #l7 = [c**2 for c in l6]
            #l8 = [d.any() for d in l7]
            #print(l6[-1], l7[-1], l8[-1])
            #print(any(l8))
            #return test


            #tmpt = rref1.to_here() # blocking

            #results.append(rref1.to_here())
            #if i == 0:
            #    result = tmpt
            #else:
            #    result += tmpt

            #tmpt = rpc_contract(self.contraction_list, new_arrays)
            #results.append(test)

        #  Use the blocking API to_here() to retrieve the result value locally
        results.extend([r.to_here() for r in remote_results])

        #print(results)
        result = sum(results)
                
        # rpc.shutdown()
        #print("cyc")
        return result



    def RPC_parallel_sliced_contract_GPU(self, arrays):
        from .distributed import rpc_contract_GPU
        import torch.distributed.rpc as rpc
        import os
        world_size = int(os.environ['world_size'])
        rank = int(os.environ['rank'])
        gpus_per_cpu = int(os.environ['gpus_per_cpu'])
        total_size = (world_size-1) * gpus_per_cpu # master node is not supposed to be used for calculation.
        
        if total_size == 0:
            raise ValueError("No GPU is assigned for distributed_GPU parallelism!!")

        if world_size < 2:
            raise ValueError("world_size must larger than 2!!")
        if rank != 0:
            raise ValueError("Master must be in rank 0!!")

        if self.num_slices > total_size:
            warnings.warn("Total number of slices is larger than total number of GPUs")

        list_arrays = [[] for _ in range(world_size-1)]

        for i in range(self.num_slices):

            # get the corresponding sliced arrays
            new_arrays = self.get_sliced_arrays(arrays, i)

            which_rank = i%(world_size-1) + 1
            list_arrays[which_rank - 1].append(new_arrays) # count from 0

        results = []
        for j in range(1, world_size): # neglect the master computer

            # put data into corresponding thread
            which_rank = j
            name = f"worker_{which_rank}"
            #print(name)

            # do the contraction
            rref1 = rpc.remote(name, rpc_contract_GPU, args=(self.contraction_list, list_arrays[which_rank - 1]))
            #print(tmpt)

            #tmpt = rref1.to_here() # blocking

            results.append(rref1)
            #if i == 0:
            #    result = tmpt
            #else:
            #    result += tmpt

        #  Use the blocking API to_here() to retrieve the result value locally
        results = [r.to_here() for r in results]
        result = sum(results)
                
        # rpc.shutdown()
        #print("cyc")
        return result



    # pylint: disable=invalid-name
    def GPU_parallel_sliced_contract(self, arrays, backend = 'torch'):
        r'''
        using GPUs to do parallel contraction for sliced arrays.
        '''

        import torch  # pylint: disable=import-outside-toplevel

        gpus_count = int(torch.cuda.device_count())

        if not gpus_count:
            raise ValueError("There's no GPU in this computer! please use cpu mode!")

        available_gpus = [torch.device('cuda:'+''.join(str(i))) for i in range(gpus_count)]  # pylint: disable=no-member
        host_device = arrays[0].device
        #print(available_gpus)
        #print(host_device)
        #print(type(available_gpus[0]))
        #print(type(host_device))

        results = []
        for i in range(self.num_slices):

            # get the corresponding sliced arrays
            new_arrays = self.get_sliced_arrays(arrays, i)

            # put data into corresponding GPU
            which_gpu = i%gpus_count
            GPU_arrays = [array.to(available_gpus[which_gpu]) for array in new_arrays]
            #print([ix.shape for ix in new_arrays])

            # do the contraction
            tmpt = core_contract(self.contraction_list, GPU_arrays, backend = backend)
            #print(tmpt)

            results.append(tmpt)

            # transfer the data back to host device
            # all the data must be on the same device and it should be the same as input data device
            #tmpt = tmpt.to(host_device)

            #if i == 0:
            #    result = tmpt
            #else:
            #    result += tmpt
        results = [r.to(host_device) for r in results]
        result = sum(results)
        return result  


    def single_thread_sliced_contract(self, arrays, backend = 'jax'):
        r'''
        using single CPU thread to do the contraction for all the sliced arrays.
        '''
        #if(backend == 'torch'):
        #    for array in arrays:
        #        print(array.device)
        for i in range(self.num_slices):
            new_arrays = self.get_sliced_arrays(arrays, i)
            #print([ix.shape for ix in new_arrays])
            tmpt = core_contract(self.contraction_list, new_arrays, backend = backend)
            #print("cyc: ", tmpt)
            if i == 0:
                result = tmpt
            else:
                result += tmpt
        return result







def core_contract(contraction_list, arrays, backend = 'jax'):
    r"""
    Contract the result without slice.

    Args:
        arrays(array): elements in the tensors to be contracted.
        backend(String): Computation backend: jax or pytorch
    """
    if backend == 'jax':  # pylint: disable=no-else-return
        #import jax
        from jax import numpy as jnp  # pylint: disable=import-outside-toplevel

        if arrays[0].shape == (2,):
            if arrays[0][0] == 1. / jnp.sqrt(2.):
                contraction_list.append([28, 7, 4, 13, 50, 0, 14, 2, 7, 4, 13, 6])

        result = jax_core_contract(contraction_list, arrays)

        return result


    elif backend == 'torch':
        import torch  # pylint: disable=import-outside-toplevel
        #from torch import tensor

        #print("arrays: ", arrays)
        for contraction in contraction_list[:-1]:#last one is transpose for final result correction
            order_operand, do_einsum, einsum_str_or_axes = contraction
            # ATTENTION!! MUST pop in order
            temp_operands = [arrays.pop(x) for x in order_operand]

            #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape)
            #print(temp_operands)
            
            if do_einsum:
                #print(einsum_str_or_axes)
                #print(temp_operands)
                result = torch.einsum(einsum_str_or_axes[0], *temp_operands)
            else:
                result = torch.tensordot(*temp_operands, dims=einsum_str_or_axes[0])

            #print(result)
            arrays.append(result)
            #print(arrays)

        final_transpose = contraction_list[-1]
        #print(arrays[0], final_transpose)
        result = arrays[0].permute(final_transpose)
        return result

    else:
        raise ValueError(f'Unknow backend {backend}')




def jax_core_contract(contraction_list, arrays):
    r"""
    Contract the result without slice.

    Args:
        arrays(array): elements in the tensors to be contracted.
    """

    from jax import numpy as jnp  # pylint: disable=import-outside-toplevel

    if arrays[0].shape == (2,):
        if arrays[0][1] == 1. / jnp.sqrt(2.):
            check = contraction_list.pop(-1)
            _egg = []
            for cc in check:
                _symbol = _EINSUM_SYMBOLS_BASE[cc]
                _egg.append(_symbol)
                _s_egg = ''.join(_egg)
            raise ValueError(f'{_s_egg} remind you, please change the tn_simplify value!')
    
    for contraction in contraction_list[:-1]:#last one is transpose for final result correction
        order_operand, do_einsum, einsum_str_or_axes = contraction
        # ATTENTION! MUST pop in order!!
        temp_operands = [arrays.pop(x) for x in order_operand]

        if do_einsum:
            #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape)
            #print(temp_operands)
            result = jnp.einsum(einsum_str_or_axes[0], *temp_operands)
            #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape, result.shape)
        else:
            #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape)
            #print(temp_operands)
            result = jnp.tensordot(*temp_operands, axes=einsum_str_or_axes[0])

        #print(einsum_str_or_axes[0], temp_operands[0].shape, temp_operands[1].shape, result.shape)
        arrays.append(result)

    final_transpose = contraction_list[-1]
    result = jnp.transpose(arrays[0], final_transpose)
    return result

#core_contract_ray = RAY.remote(core_contract)
        

# pylint: disable=pointless-string-statement
'''
def trial_fn(self, inputs, output, size_dict, vertices_to_remove, **kwargs):
    abc, root = self(inputs, output, size_dict, vertices_to_remove, **kwargs)
    return abc, root


def cyc(graph, remove_N, *operands, **kwargs):
    repeated_times = kwargs.pop('repeated_times', 10)
    edges_to_remove = kwargs.pop('edges_to_remove', [])
    prev_remove = set(edges_to_remove)

    input_list = operands[0]
    output = operands[1]
    size_dict = operands[2]

    input_set = set(input_list)
    output_set = set(output)
    size_dict = {}

    for tnum, term in enumerate(input_list):
        sh = input_shapes[tnum]
        for cnum, char in enumerate(term):
            dim = int(sh[cnum])
            if char in size_dict:
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError("")
            else:
                size_dict[char] = dim
    tw = None
    for _ in range(repeated_times):
        kahypar_to_tree = PartitionTreeBuilder()
        char_order, root = kahypar_to_tree.trial_fn(input_sets, output_set, size_dict,edges_to_remove)

        set_prev_vertices(root)
        set_whole_vertices(root)

        bitr_tw, rabbishstw = set_values(root)
        number, stw = get_edges_to_remove(root, bitr_tw, remove_N, prev_remove)



def set_prev_vertices(root):
    if root.left is not None:
        root.left.prev_vertices = root.prev_vertices | root.vertices
        root.right.prev_vertices = root.prev_vertices | root.vertices
        set_prev_vertices(root.left)
        set_prev_vertices(root.right)

def set_whole_vertices(root):
    for ts in reversed(root.levels):
        for t in ts:
            if t.left is None:
                t.whole_vertices = t.vertices
            elif t.left is not None:
                t.whole_vertices = t.left.whole_vertices | t.right.whole_vertices

def set_values(root):
    tw = 0
    for ts in reversed(root.levels):
        for t in ts:
            all_vertices = t.whole_vertices & t.prev_vertices | t.vertices
            _value = len(all_vertices)
            t.value = _value
            if _value>tw:
                stw = all_vertices
                tw = _value
    return tw, list(stw)
'''

# pylint: disable=pointless-string-statement
'''
def get_edges_to_remove(root, best_tw, remove_N, prev_remove):
    number = 0
    all_vertices = set()
    vertices_for_return = set()
    while 1:
        if number > remove_N:
            break

        i = 0
        for ts in reversed(root.levels):
            for t in ts:
                if t.value >= best_tw - number:
                    all_ = t.whole_vertices & t.prev_vertices | t.vertices
                    if i == 0:
                        all_vertices = all_vertices
                    else:
                        all_vertices = all_ & all_vertices
                    i += 1

        all_vertices -= prev_remove

        if len(all_vertices) < number:
            break

        vertices_for_return = all_vertices
        number += 1

    return number -1, list(vertices_for_return)
'''




