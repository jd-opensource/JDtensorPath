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
Helper functions
'''

import collections
# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods

# pylint: disable=invalid-name
def separate(xs, blocks):
    r'''
    seperate hyper-graph's nodes into parts according to kahypar output

    **Example**
    
    .. code-block:: python3
        xs = ['a', 'b', 'c', 'd']
        blocks = [0, 1, 0, 2]
        >>> separate(xs, blocks)
        [['a', 'c'], ['b'], ['d']]
    '''
    sorter = collections.defaultdict(list)
    for x, b in zip(xs, blocks):  # pylint: disable=invalid-name
        sorter[b].append(x)
    x_b = list(sorter.items())
    x_b.sort()
    return [x[1] for x in x_b]

def count_flops(eq, size_dict):
    r'''
    calculate the flops of matrix product.
    A: m*p
    B: p*n
    C: A*B = m*n
    the flops should be m*n*(2*p-1)
    C: AB = m*p*n
    the flops should be m*n*p

    **Example**
    
    .. code-block:: python3
        size_dict = {'m':2, 'p':3, 'n':4}
        eq = ({'m', 'p'}, {'p', 'n'}, {'m', 'n'})
        >>> count_flops(eq, size_dict)
        40

    '''
    (m_a, m_b, m_c) = eq
    p = (m_a | m_b) - m_c

    size_mn = compute_size_by_dict(m_c, size_dict)
    size_p = compute_size_by_dict(p, size_dict)

    flops = size_mn*(2*size_p -1)
    return flops

def compute_size_by_dict(indices, size_dict):
    r'''
    Computes the product of the elements in indices based on the dictionary.

    **Example**
    
    .. code-block:: python3
        indices = ['a', 'b']
        size_dict = {'a':2, 'b':3}
        >>> compute_size_by_dict(indices, size_dict)
        6

    '''
    size = 1
    for idx in indices:
        size *= size_dict[idx]

    return size

def dec_to_bin(x, size):
    r'''
    converting a decimal number into binary format.

    **Example**

    .. code-block:: python3
        x = 3
        size = 2
        >>> dec_to_bin(3, 2)
        [1, 1]
    '''
    # pylint: disable=invalid-name
    import math
    size = int(math.log2(size))
    n = bin(x)[2:]
    n = n.zfill(size)
    result = [int(ix) for ix in n]
    return result

def to_dynamic_base(x, bases):
    r"""
    Transfer the decimal integer ``x`` with respect to the 'dynamical' ``bases``.
    This function can be used to enumerate all the combination of different
    index values.

    Examples
    --------
        >>> to_dynamic_base(5, [2, 2, 2])  # to binary
        [1, 0, 1]
        >>> to_dynamic_base(5783, [10, 10, 10, 10])  # to decimal
        [5, 7, 8, 3]

        >>> base = [4, 7, 9, 8, 5, 7, 2, 1, 3]  # to arbitrary bases
        >>> for x in range(397294, 397305):
        ...     print(to_dynamic_base(x, base))
        [3, 5, 2, 3, 4, 2, 1, 0, 1]
        [3, 5, 2, 3, 4, 2, 1, 0, 2]
        [3, 5, 2, 3, 4, 3, 0, 0, 0]
        [3, 5, 2, 3, 4, 3, 0, 0, 1]
        [3, 5, 2, 3, 4, 3, 0, 0, 2]
        [3, 5, 2, 3, 4, 3, 1, 0, 0]
        [3, 5, 2, 3, 4, 3, 1, 0, 1]
        [3, 5, 2, 3, 4, 3, 1, 0, 2]
        [3, 5, 2, 3, 4, 4, 0, 0, 0]
        [3, 5, 2, 3, 4, 4, 0, 0, 1]
        [3, 5, 2, 3, 4, 4, 0, 0, 2]

    """
    base_size = [prod(bases[i + 1:]) for i in range(len(bases))]
    new_x = []
    for sz in base_size:
        num = x // sz
        new_x.append(num)
        x -= num * sz
    return new_x

def prod(seq):
    r"""
    Compute the product of a sequence of numbers.

    **Example**

    .. code-block:: python3
        seq = [2, 3]
        >>> prod(seq)
        6

    """
    ret = 1
    for x in seq:
        ret *= x
    return ret


def alter(file, dict_str_old_new):
    """
    Replace the string inside a file
    :param file: file name
    :param dict_str_old_new: dict of old str to new str
    
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            for old_str, new_str in dict_str_old_new.items():
                if old_str in line:
                    #print(line)
                    line = new_str
            file_data += line
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)