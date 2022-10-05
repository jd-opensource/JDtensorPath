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

r'''
Generate trials for possible contraction order and slicing.
'''

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods
# pylint: disable=too-many-arguments

from copy import deepcopy
import random

from jdtensorpath.ray_parallel import get_ray, read_out_ray
from .build_tree import jd_partition
from .helpers import compute_size_by_dict

#RAY = get_ray()

def generate_trials_parallel(num_cpus, trial_times, inputs, size_dict, output = None, precut_edges = None, imbalance=0.01, parent_node = None, **kwargs):
    r'''
    generate trials for contraction order parallelly
    '''

    #print("generate_trials_parallel start")
    ray = get_ray(num_cpus)
    #print("generate_trials_parallel get ray")
    num_cpu = int(ray.available_resources()['CPU'])
    #print(num_cpu)
    get_trial_ray = ray.remote(get_trial)

    ids_list = []
    for _ in range(trial_times):
        ids_list.append(get_trial_ray.remote(inputs, size_dict, output, precut_edges, imbalance, parent_node, **kwargs))
        if len(ids_list) > num_cpu + 3:
            trial = read_out_ray(ids_list)
            #print("size: ", trial['size'])
            yield trial
    while len(ids_list) > 0:
        trial = read_out_ray(ids_list)
        #print("size: ", trial['size'])
        yield trial
    #trials = ray.get(ids_list)
    #return trials

def get_trial(inputs, size_dict, output = None, precut_edges = None, imbalance=0.01, parent_node = None, **kwargs):
    r"""
    Each trial to cut

    jd_partition -> binary tree of (tree in compiled_circuit)

    current_leaves-> special
    """
    
    #print("start gen_trials's get_trial")
    precut_edges = list(precut_edges) # deepcopy precut_edge
    current_leaves = []
    binary_tree = jd_partition(
        current_leaves, inputs, size_dict, output=output, precut_edges=precut_edges, imbalance=imbalance, parent_node=parent_node, **kwargs
        )
    size = binary_tree.get_max_size(size_dict)
    flops = binary_tree.get_flops(size_dict)
    trial = {'current_leaves':current_leaves, 'binary_tree':binary_tree, 'size':size, 'flops':flops}
    #print("end gen_trials's get_trial")
    return trial


#get_trial_ray = RAY.remote(get_trial)





def generate_trials(trial_times, inputs, size_dict, output = None, precut_edges = None, imbalance=0.01, parent_node = None, **kwargs):
    r'''
    generate trials for contraction order
    '''
    #print("start gen_trials's generate_trials")
    for _ in range(trial_times):
        yield get_trial(inputs, size_dict, output, precut_edges, imbalance, parent_node, **kwargs)





# for slicing parts

def find_slice(sliced_sets, output_inds, target_size, size_dict):
    r"""
    Find the sliced_set of tensor network.

    Args:
        sliced_sets(int): TBD
        target_size(int): Target size of set after slice operations

    """
    sets = deepcopy(sliced_sets)
    #print(sets)
    removed = set()
    trial = {}

    while 1:

        choice = random.randint(0,1)
        if choice:
            rm = remove_most_intersect(sets, output_inds)
        else:
            rm = remove_most_larges(sets, output_inds)

        removed = removed | rm

        sizes =[compute_size_by_dict(inds, size_dict) for inds in sets]
        size = max(sizes)

        if size < target_size + 1:
            break

    num_slices = compute_size_by_dict(removed, size_dict)
    trial['num_slices'] = num_slices
    trial['removed'] = removed
    trial['size'] = size
    return trial



def generate_slicing_trials_parallel(num_cpus, trial_times, sliced_sets, output_inds, target_size, size_dict):
    r'''
    generate slicing trials parallelly
    '''

    ray = get_ray(num_cpus)
    num_cpu = int(ray.available_resources()['CPU'])
    #print(num_cpu)
    find_slice_ray = ray.remote(find_slice)

    ids_list = []
    for _ in range(trial_times):
        ids_list.append(find_slice_ray.remote(sliced_sets, output_inds, target_size, size_dict))
        if len(ids_list) > num_cpu + 3:
            yield read_out_ray(ids_list)
    while len(ids_list) > 0:
        yield read_out_ray(ids_list)
    #trials = ray.get(ids_list)
    #return trials



#find_slice_ray = RAY.remote(find_slice)




def generate_slicing_trials(trial_times, sliced_sets, output_inds, target_size, size_dict):
    r'''
    generate slicing trials.
    '''

    for _ in range(trial_times):
        yield find_slice(sliced_sets, output_inds, target_size, size_dict)




def remove_most_intersect(sets, output_inds):
    r'''
    remove indice from width

    **Example**

    .. code-block:: python3

        sets = [
                {'a', 'b', 'c', 'd'}, 
                {'a', 'e', 'f'},
                {'c', 'g'},
                {'c', 'h'}
                ]            

    >>> remove_most_intersect(sets)
    {'c'}

    '''
    indices_list = []
    #print(sets)
    for indices in sets:
        indices_list.extend(list(indices))

    # except output indices    
    indices_list = [index for index in indices_list if index not in output_inds]
    #print(indices_list)
    if indices_list:
        removed = set(max(indices_list, key=indices_list.count))
    else:
        # cannot find, put a empty set
        removed = set()
    for i, _ in enumerate(sets):
        sets[i] = sets[i] - removed
    return removed


def remove_most_larges(sets, output_inds):
    r'''
    remove indice from top

    **Example**

    .. code-block:: python3

        sets = [
                {'a', 'b', 'c', 'd'}, 
                {'a', 'e', 'f'},
                {'c', 'g'},
                {'c', 'h'}
                ]            

    >>> remove_most_larges(sets)
    {'a'}

    '''
    sets.sort(key=lambda indices:-1*len(indices))
    set_output = set(output_inds)
    # except output indices
    intersect = sets[0] - set_output
    removed = set()
    for indices in sets[1:]:
        previous_set = set(intersect)
        # except output indices
        intersect = intersect & (indices - set_output)
        if len(intersect) == 0:
            # randomly pip one index
            # prevent pop from empty set
            try:
                removed = set(previous_set.pop())
            except KeyError:
                removed = set()
            break
    for i, _ in enumerate(sets):
        sets[i] = sets[i] - set(removed)
    return removed
