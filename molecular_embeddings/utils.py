from collections.abc import Iterable
import os
import numpy as np
import time
np.set_printoptions(formatter={'float': '{:0.2f}'.format})
from functools import wraps
from typing import Callable
def Apply(l: list, func: Callable):
    return list(map(func, l))

def Union(l1: list, l2:list):
    return list(set(l1).union(set(l2)))

def Intersection(l1: list, l2:list):
    return list(set(l1).intersection(set(l2)))

def Difference(l1: list, l2:list):
    return list(set(l1).difference(set(l2)))

def IoU(l1: list, l2:list):
    return len(Intersection(l1, l2)) / len(Union(l1, l2))
                                                    
def flatten(x):
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
    
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds') # {args} {kwargs}
        return result
    return timeit_wrapper

def WHERE(list, operator, threshold):
    return np.where(operator(np.array(list), threshold))[0]


def get_newest_file(folder, *args):
    file_paths = [
        os.path.join(folder, file) for file in os.listdir(folder) if all(arg in file for arg in args)
    ]
    assert file_paths, f"There are no files containing {args}"
    return max(file_paths, key=os.path.getctime)

def get_updated_dic(dict1, dict2):
    dict1.update(dict2)
    return dict1

if __name__ == '__main__':
    print(get_newest_file('data/', 'cyp', '2', 'ca'))