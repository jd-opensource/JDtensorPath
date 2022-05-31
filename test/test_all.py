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


r"""
Test file for storage and circuit.
"""

from test_JD_opt_tn import testTree, testHyperGraph, testSlicing, testHelpers

def run_the_test():
	"""
	"""

	print(" ")
	print("Star testing hypergraph function!")
	print(" ")
	a = testHyperGraph()
	a.test_construction()
	a.test_properties()
	print(" ")
	print("Finish hypergraph function test, everything ok!")
	print(" ")
	print(" ")

	print(" ")
	print("Star testing contraction tree function!")
	print(" ")
	a = testTree()
	a.test_construction()
	a.test_get_all_nodes()
	print(" ")
	print("Finish contraction tree function test, everything ok!")
	print(" ")
	print(" ")
	
	print(" ")
	print("Star testing slicing function!")
	print(" ")
	a = testSlicing()
	a.test_remove_most_intersect()
	a.remove_most_larges()
	print(" ")
	print("Finish slicing function test, everything ok!")
	print(" ")
	print(" ")

	print(" ")
	print("Star testing JD_opt_tn helpers function!")
	print(" ")
	a = testHelpers()
	a.test_separate()
	a.test_count_flops()
	a.test_compute_size_by_dict()
	a.test_dec_to_bin()
	a.test_to_dynamic_base()
	a.test_prod()
	print(" ")
	print("Finish JD_opt_tn helpers function test, everything ok!")
	print(" ")
	print(" ")

	print("All the tests pass! congratulation! Now you can play with JDtensorPath together with TeD-Q, good luck~!")