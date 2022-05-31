## JDtensorPath
A tensor contraction path finder. Tensor network can be contracted in a specific order, JDtensorPath utilizes hypergraph partitioning to build contraction order. 

## Main Features
* ***Hyper-graph based path finder.*** Search a tensor network contraction path by using hyper-graph partitioning technique.
* ***Fast and memory saving.*** Using new algorithm in setting partition weight and slcing, both memory usage and time consumption is smaller that other softwares.
* ***Good compatibility.*** Can use jax or pytorch backend to carry out the contraction.
* ***parallelism.*** Contraction can be done in a parallel manner.

## Installation
#### Prerequisite
```
pip install tedq kahypar==1.1.6
```
#### Install
```
pip install -e .
```

## Getting started
Use JDtensorPath together with [TeD-Q](https://github.com/JDEA-Quantum-Lab/TeD-Q).
### Simple example
#### Define the circuit with TeD-Q framework
```
import tedq as qai
def circuitDef(params):
    qai.RX(params[0], qubits=[0])
    qai.RY(params[1], qubits=[0])
    return qai.expval(qai.PauliZ(qubits=[0]))
```
#### Quantum circuit constuction
```
number_of_qubits = 1
parameter_shapes = [(2,)]
my_circuit = qai.Circuit(circuitDef, number_of_qubits, parameter_shapes = parameter_shapes)
```
#### Compile quantum circuit with JDtensorPath
```
# 'target_num_slices' is useful if you want to do the contraction in parallel, it will devide the tensor network into pieces and then calculat them in parallel
# 'math_repeats' means how many times are going to run JDtensorPath to find a best contraction path
# 'search_parallel' means to run the JDtensorPath in parallel, True means to use all the CPUs, integer number means to use that number of CPUs
from jdtensorpath import JDOptTN as jdopttn
slicing_opts = {'target_size':2**28, 'repeats':500, 'target_num_slices':None, 'contract_parallel':False}
hyper_opt = {'methods':['kahypar'], 'max_time':120, 'max_repeats':12, 'search_parallel':True, 'slicing_opts':slicing_opts}
my_compilecircuit = circuit.compilecircuit(backend="pytorch", use_jdopttn=jdopttn, hyper_opt = hyper_opt, tn_simplify = False)
```
#### Run the circuit
```
import torch
a = torch.tensor([0.54], requires_grad= True)
b = torch.tensor([0.12], requires_grad= True)
my_params = (a, b)
c = my_compilecircuit(*my_params)
>>> c = tensor([0.8515], grad_fn=<TorchExecuteBackward>)
```
### Learn more
Please refer to the tutorials and examples of TeD-Q.

## Tutorial and examples
For more diverse examples of using JDtensorPath together with TeD-Q, please refer to the following tutorials or our official [documentation](https://tedq.readthedocs.io) website.
#### [1D Many Body Localization](/examples/Many_body_Localization_1D.ipynb)

## Authors
JDtensorPath is released by JD Explore Academy and is currently maintained by [Xingyao Wu](https://github.com/xywu1990). The project is not possible without the efforts made by our [contributors](https://github.com/JDEA-Quantum-Lab/jd-tensor-path/graphs/contributors).

## License
JDtensorPath is free and open source, released under the Apache License, Version 2.0.
