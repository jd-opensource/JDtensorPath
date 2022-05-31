#   Copyright 2021-2024 Jingdong Digits Technology Holding Co.,Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#	   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


r"""
Test file for jdtensorpath.
"""
import numpy as np

from jdtensorpath.build_tree import TreeNode, HyperGraph
from jdtensorpath.helpers import *
from jdtensorpath.gen_trials import remove_most_intersect, remove_most_larges


class testTree():
	r'''
	'''
	def test_construction(self):
		r'''
		'''
		tree = TreeNode()
		assert isinstance(tree, TreeNode)

		print("test tree construction ok!")

	def test_get_all_nodes(self):
		r'''
			   node4
			   /
			node3
			/  \
		node1 node2
		'''
		node1 = TreeNode()
		node2 = TreeNode()
		node3 = TreeNode(node1, node2)
		node4 = TreeNode(node3)

		nodes = node4.get_all_nodes()

		assert nodes[0] == node4
		assert nodes[1] == node3
		assert nodes[2] == node1
		assert nodes[3] == node2

		print("test tree get all nodes function ok!")

class testHyperGraph():
	r'''
	'''
	def test_construction(self):
		r'''
		'''
		inputs = [['a'], ['b'], ['a', 'c'], ['b', 'd'], ['c', 'd', 'e', 'f']]  # 5 nodes, 6 edges
		output = ['e', 'f']  # output indices
		size_dict = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6}  # edges weighting
		hg = HyperGraph(inputs, size_dict, output)

		assert isinstance(hg, HyperGraph)

		print("test HyperGraph construction ok!")

	def test_properties(self):
		r'''
		'''
		inputs = [['a'], ['b'], ['a', 'c'], ['b', 'd'], ['c', 'd', 'e', 'f']]  # 5 nodes, 6 edges
		output = ['e', 'f']  # output indices
		size_dict = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6}  # edges weighting
		hg = HyperGraph(inputs, size_dict, output)

		num_nodes = hg.get_num_nodes()
		num_edges = hg.get_num_edges()
		edge_weights = hg.get_edge_weights()
		output_nodes = list(hg.output_nodes())

		assert num_nodes == 5
		assert num_edges == 6
		assert edge_weights == (1, 2, 3, 4, 5, 6)
		assert output_nodes == [4]

		print("test HyperGraph properties ok!")

class testSlicing():
	r'''
	'''
	def test_remove_most_intersect(self):
		r'''
		'''
		sets = [
				{'a', 'b', 'c', 'd'}, 
				{'a', 'e', 'f'},
				{'c', 'g'},
				{'c', 'h'}
				]

		rm = remove_most_intersect(sets)

		assert rm == {'c'}

		print("test slicing remove most intersect ok!")

	def remove_most_larges(self):
		r'''
		'''
		sets = [
				{'a', 'b', 'c', 'd'}, 
				{'a', 'e', 'f'},
				{'c', 'g'},
				{'c', 'h'}
				]

		rm = remove_most_larges(sets)

		assert rm == {'a'}

		print("test slicing remove most larges ok!")

class testHelpers():
	r'''
	'''
	def test_separate(self):
		r'''
		'''
	xs = ['a', 'b', 'c', 'd']
	blocks = [0, 1, 0, 2]
	members = separate(xs, blocks)
	correct = [['a', 'c'], ['b'], ['d']]

	assert members == correct

	def test_count_flops(self):
		r'''
		'''
		size_dict = {'m':2, 'p':3, 'n':4}
		eq = ({'m', 'p'}, {'p', 'n'}, {'m', 'n'})
		flops = count_flops(eq, size_dict)

		assert flops == 40

	def test_compute_size_by_dict(self):
		r'''
		'''
		indices = ['a', 'b']
		size_dict = {'a':2, 'b':3}
		size = compute_size_by_dict(indices, size_dict)

		assert size == 6

	def test_dec_to_bin(self):
		r'''
		converting a decimal number into binary format.
		'''
		x = 3
		size = 2
		new_x = dec_to_bin(3, 2)
		assert new_x == [1, 1]
	
	def test_to_dynamic_base(self):
		r'''
		'''
		base = [4, 7, 9, 8, 5, 7, 2, 1, 3]
		x = 397294
		new_x = to_dynamic_base(x, base)

		assert new_x == [3, 5, 2, 3, 4, 2, 1, 0, 1]

	def test_prod(self):
		r'''
		Compute the product of a sequence of numbers.
		'''
		seq = [2, 3]
		size = prod(seq)

		assert size == 6

		print("test all helpers function ok!")
