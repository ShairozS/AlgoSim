# Program Search using Similarity

A companion repo to the paper "Towards a Measure of Algorithm Similarity".

### Research Questions
- [ ] Is it possible to do unsupervised clustering of algorithms based on a similarity metric?
- [ ] What is the variance of algorithms generated form SOTA LLM models (based on this metric)?
- [ ] Can we bias LLM sampling to generate original algorithms?

This repo provides a set of tools to compare algorithms written in Python, in order to discover meaningful differences and/or filter for potentially novel algorithms. To get started with the repo, we first need to have the algorithms we'd like to compare implemented as (pure) Python functions. From there, a simple 3-step process:


1) Turn the implemented algorithms into self-contained Python functions
2) Turn the functions into CompiledFunctions
3) Define an input set to run the comparisons against
4) Utilize the .compare(), .similarity(), and .emoc() methods of each CompiledFunction to analyze which algorithms are likely to be the same.


## Example Usage: Comparing Sorting Algorithms
For example, consider the following (2) implementations of Bubble Sort:

```python
def bubble_sort1(arr):
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def bubble_sort2(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
```

The main difference between these two is the use of a "swapped" flag to allow early termination. Now, consider an implementation of Merge Sort:


```python

def quicksort(arr):
    """
    Sorts a list using the Quicksort algorithm.
    """
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]  # Choose the middle element as pivot
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)

```

To utilize our algorithm similarity metric, we first turn each of these into CompiledFunction objects:

```python
from algosim.utils import CompiledFunction

bsort1 = CompiledFunction(bubble_sort1)
bsort2 = CompiledFunction(bubble_sort2)
qsort = CompiledFunction(quicksort)

```
Then we simply do pairwise comparison:

```python
print(f"Bubble Sort 1 vs. Bubble Sort 2: {bsort1.compare(bsort2)}")
print(f"Bubble Sort 1 vs. Quicksort: {bsort1.compare(qsort)}")
print(f"Bubble Sort 2 vs. Quicksort: {bsort2.compare(qsort)}")
```

## Example: Comparing LLM Generated Algorithms

