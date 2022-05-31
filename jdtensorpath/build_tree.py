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
Module for building contraction tree.
'''

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods

import random
import collections
from os.path import join, abspath, dirname
from copy import deepcopy
from toolz import unique
import kahypar

from .helpers import separate, count_flops, compute_size_by_dict


KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)), 'kahypar_profiles')

# pylint: disable = pointless-string-statement
r'''
Precut_edge ==> better than cotengra
'''


# pylint: disable=too-many-arguments, too-many-locals
def jd_partition(current_leaves, inputs, size_dict, output = None, precut_edges = None, imbalance=0.01, parent_node = None, **kwargs):
    r'''
    Cursively partition the hyper graph into a contraction tree.
    '''
    if output is None:
        output = []
    if precut_edges is None:
        precut_edges = []

    #return 1
    #print("start build_tree's jd_partition")
    subgraph = inputs
    subsize = len(subgraph)

    local_root = TreeNode()
    local_root.parent = parent_node

    if subsize == 1:
        local_root.value = 2 #if tensor of childs is valid :2 one:1 zero:0
        local_root.outer_indices = subgraph[0]
        local_root.outer_indices_set = set(subgraph[0])
        local_root.size = len(local_root.outer_indices_set)
        current_leaves.append(local_root)
        return local_root

    precut_set = set(precut_edges)

    temp_set = precut_set | set(output) # output: indice of final state
    for tensor in subgraph:
        local_root.outer_indices_set = local_root.outer_indices_set | set(tensor)
    local_root.outer_indices_set = local_root.outer_indices_set & temp_set
    local_root.size = len(local_root.outer_indices_set)
    #print(local_root.outer_indices)

    
    
    if subsize>5:
        membership = kahypar_partition(subgraph, size_dict, output, precut_edges, imbalance*random.random(), **kwargs)
        N_group = set(membership)  # pylint: disable=invalid-name
        if len(N_group) != 2:
            membership = [0 for _ in range(subsize//2)]
            membership.extend([1 for _ in range(subsize - subsize//2)])

    else:
        membership = [0 for _ in range(subsize//2)]
        membership.extend([1 for _ in range(subsize - subsize//2)])            

    new_subgs = tuple(separate(subgraph, membership))
    set_a = set()
    for tensor in new_subgs[0]:
        set_a = set_a | set(tensor)
    set_b = set()
    for tensor in new_subgs[1]:
        set_b = set_b | set(tensor)
    intersect = set_a & set_b
    precut_edges = list(precut_set | intersect)
    imbalance = 1- 32./subsize
    imbalance = max(imbalance, 0.01)
    imbalance = min(imbalance, 0.5)


    local_root.left = jd_partition(current_leaves, new_subgs[0], size_dict, output, precut_edges, imbalance, local_root, **kwargs)

    local_root.right = jd_partition(current_leaves, new_subgs[1], size_dict, output, precut_edges, imbalance, local_root, **kwargs)
    

    #print("end build_tree's jd_partition")
    return local_root


# pylint: disable=c-extension-no-member 
def kahypar_partition(
    inputs,
    size_dict,
    output=None,
    precut_edges=None,
    imbalance=0.01,
    mode='direct',
    objective='cut',
    quiet=True,
):
    r'''
    use kahypar to do the partitioning.
    '''

    if output is None:
        output = []
    if precut_edges is None:
        precut_edges = []
        
    #print("start build_tree's kahypar_partition")
    seed = random.randint(0, 2**31 - 1)    

    parts = 2    

    hgoutput = deepcopy(output)
    hgoutput = hgoutput.extend(precut_edges)
    hg = HyperGraph(inputs, size_dict, hgoutput)  # pylint: disable=invalid-name   



    index_vector, edge_vector = hg.get_hyperedges()

    hypergraph_kwargs = {
        'num_nodes': hg.get_num_nodes(),
        'num_edges': hg.get_num_edges(),
        'index_vector': index_vector,
        'edge_vector': edge_vector,
        'k': parts,
    }    


    hypergraph_kwargs['edge_weights'] = hg.get_edge_weights()
    hypergraph_kwargs['node_weights'] = hg.get_node_weights()

    hypergraph = kahypar.Hypergraph(**hypergraph_kwargs)    


    profile_mode = {'direct': 'k', 'recursive': 'r'}[mode]
    
    profile = f"{objective}_{profile_mode}KaHyPar_sea20.ini"


    context = kahypar.Context()

    #print("here? before kahypar profile")
    context.loadINIconfiguration(join(KAHYPAR_PROFILE_DIR, profile))
    #print("here? after kahypar profile") 

    context.setK(parts)
    context.setSeed(seed)
    context.suppressOutput(quiet)
    context.setEpsilon(imbalance * parts)

    #print("here? before kahypar partition")
    #print("kahypar profile location:  ", KAHYPAR_PROFILE_DIR, "  ", profile)
    kahypar.partition(hypergraph, context)

    #print("end build_tree's kahypar_partition")
    return [hypergraph.blockID(i) for i in hypergraph.nodes()]



class TreeNode:
    r'''
    a node of contraction tree.
    
    **Example:**
    
        >>> node1 = TreeNode()
        >>> node2 = TreeNode()
        >>> node3 = TreeNode(node1, node2)
        >>> node4 = TreeNode(node3)
        Then it will have this structure:
                 node4
                  /
               node3
               /   \
            node1 node2
    '''

    def __init__(self, left = None, right = None, parent = None):
        self.value = 0# indicating how many child nodes has been ready
        # this will be filled on get_contraction_order() function
        self.outer_indices = []
        self.outer_indices_set = set()
        self.size = None# size of the tensor
        self.left = left
        self.right = right
        self.parent = parent

    # pylint: disable=modified-iterating-list
    def get_all_nodes(self):
        r'''
        return all the nodes.
        '''
        all_nodes = [self]

        for node in all_nodes:
            if node.left is not None:
                all_nodes.append(node.left)
            if node.right is not None:
                all_nodes.append(node.right)

        return all_nodes

    # pylint: disable=modified-iterating-list
    def get_max_size(self, size_dict):
        r'''
        return the max size.
        '''

        max_size = 0
        all_nodes = self.get_all_nodes()

        for node in all_nodes:
            # here must use set, since node.outer_indices is not filled yet
            indices = node.outer_indices_set
            size = compute_size_by_dict(indices, size_dict)
            if size > max_size:
                max_size = size

        return max_size

    # pylint: disable=modified-iterating-list
    # TODO
    def get_flops(self, size_dict, removed = None):
        r'''
        return flops.
        '''
        flops = 0.
        all_nodes = self.get_all_nodes()

        for node in all_nodes:
            if node.left and node.right:
                # here must use set, since node.outer_indices is not filled yet
                m_c, m_a, m_b = node.outer_indices_set, node.left.outer_indices_set, node.right.outer_indices_set

                if removed:
                    m_a = {x for x in m_a if x not in removed}
                    m_b = {x for x in m_b if x not in removed}
                    m_c = {x for x in m_c if x not in removed}

                flops += count_flops((m_a, m_b, m_c), size_dict)

        return flops

    def removed_indices(self, removed):
        r'''
        remove some indices.
        '''
        #print("remove")
        new_outer_indices = [index for index in self.outer_indices if index not in removed]
        self.outer_indices = new_outer_indices
        self.outer_indices_set = {index for index in self.outer_indices_set if index not in removed}
        if self.left is not None:
            self.left.removed_indices(removed)
        if self.right is not None:
            self.right.removed_indices(removed)

    @property
    def tree_name(self):
        r'''
        binary contraction tree
        '''
        return 'binary contraction tree'

    #@property
    #def max_size(self):
    #    r'''
    #    return size of this contraction tree
    #    '''
    #    max_size = self.get_max_size()
    #    return max_size
    
    #@property
    #def flops(self):
    #    r'''
    #    return flops of this contraction tree
    #    '''
    #    flops = self.get_flops()
    #    return flops


class HyperGraph:
    r'''
    conveting quantum circuit into hyper graph, so that kahypar can use it to do the partitioning.

    **Example**

    .. code-block:: python3

        inputs = [['a'], ['b'], ['a', 'c'], ['b', 'd'], ['c', 'd', 'e', 'f']]  # 5 nodes, 6 edges
        output = ['e', 'f']  # output indices
        size_dict = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6}  # edges weighting
        hg = HyperGraph(inputs, size_dict, output)

    >>> hg.nodes
    {0: ('a',), 1: ('b',), 2: ('a', 'c'), 3: ('b', 'd'), 4: ('c', 'd', 'e', 'f')}

    >>> hg.edges
    defaultdict(tuple,
            {'a': (0, 2),
             'b': (1, 3),
             'c': (2, 4),
             'd': (3, 4),
             'e': (4,),
             'f': (4,)})

    >>> hg.get_num_nodes()
    5

    >>> hg.get_num_edges()
    6

    >>> hg.get_edge_weights()
    (1, 2, 3, 4, 5, 6)

    >>> list(hg.output_nodes())
    [4]

    '''

    def __init__(self, inputs, size_dict, output=None):

        if output is None:
            output = []

        self.inputs = inputs
        self.output = output
        self.size_dict = size_dict

        self.nodes = dict(enumerate(map(tuple, inputs)))

        self.edges = collections.defaultdict(tuple)
        for i, term in self.nodes.items():
            for e in term:  # pylint: disable=invalid-name
                self.edges[e] += (i,)



    def get_num_nodes(self):
        r'''
        return number of nodes;
        '''
        return len(self.nodes)


    def get_num_edges(self):
        r'''
        return number of edges
        '''
        return len(self.edges)

    def get_hyperedges(self):
        r'''
        return the hyperedges
        '''

        hyperedge_indices = []
        hyperedges = []
        for e in tuple(self.edges):  # pylint: disable=invalid-name
            hyperedge_indices.append(len(hyperedges))
            hyperedges.extend(self.edges[e])
        hyperedge_indices.append(len(hyperedges))

        return hyperedge_indices, hyperedges

    def get_node_weights(self):
        r'''
        get the weighting of nodes
        '''

        if self.output:  # pylint: disable=no-else-return
            weights = list(
                self.compute_size_by_dict(term)
                for term in self.nodes.values()
                )
            for ix in self.output_nodes():  # pylint: disable=invalid-name
                weights[ix] = 1000000
            weights = tuple(weights)
            return weights

        else:
            weights = tuple(
                compute_size_by_dict(term, self.size_dict)
                for term in self.nodes.values()
                )
            return weights

    def get_edge_weights(self):
        r'''
        get the weighting of an edge.
        '''
        weights = tuple(
            self.size_dict[e] for e in tuple(self.edges)
            )
        return weights

    def output_nodes(self):
        r'''
        nodes connected to output indices
        return output nodes
        '''
        result = unique(i for e in self.output for i in self.edges[e])
        #print(list(result))
        return result




