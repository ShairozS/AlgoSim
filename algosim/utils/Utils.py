from time import time, perf_counter
import os
import dis
import tracemalloc
from collections import Counter
import numpy as np
from cProfile import Profile
import cProfile
import io
import pstats
from pstats import SortKey
from types import CodeType, FunctionType
from typing import List, Set


import dis
from collections.abc import Iterable
import textwrap
import ast
from types import CodeType, FunctionType
from typing import List, Set
import tracemalloc
from time import time, perf_counter
from collections import Counter

# helpers to compute o_term
#builtin_names = set(__builtins__.keys())
ops_to_match = ['BINARY_OP', 'COMPARE_OP']

def get_fn_names_(codeobj: CodeType, context):
    names = []
    for name in codeobj.co_names:
        if name in context and name not in builtin_names:
            if isinstance(context.get(name), FunctionType):
                names.append(name)
    return names

def get_ops_(bytecode: dis.Bytecode):
    ops = [instr.opname for instr in bytecode if instr.opname in ops_to_match]
    return ops


def levenshtein_distance(s1, s2):
    m = len(s1)
    n = len(s2)
    
    # Initialize a matrix to store distances
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Fill the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Calculate distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Deletion
                                 dp[i][j - 1],      # Insertion
                                 dp[i - 1][j - 1])  # Substitution
    
    return dp[m][n]


class CompiledFunction:
    '''
    A general purpose class for testing the equality of two Python functions given the following
    constraints:

    > The functions are well-defined and halt on the given set of inputs (e.g this class provides no methods for premature
      stopping or runtime limits

    > Equality only applies over a pre-defined domain.

    > Functions do not use or modify any global variables 

    Note that analytic determination of functional equivelency is impossible, even over a pre-defined domain. This is simply
    a heauristic test with the failure cases largely pathological.

    '''

    def __init__(self, func_string, func_name = None, compiler = 'cpython', optimization_level = 1):
        '''
        Utilize a string description of a python function, a compiler, and a
        single optimization_level for the compiler representing the presence
        of certain compiler optimizations. The function provided is not compiled
        upon initialization of this class instance.

        Parameters:
            func_string (str): 
            compiler (str): Must be one of ['cpython', 'numba']
            optimization_level (int): Either 0, 1, or 2. Currentely only supported by compiler='cpython'

        '''
        self.compiler = compiler
        self.func_string = func_string
        self.func = None
        self.optimization_level = optimization_level

        if func_string.count('def') > 1 and (func_name is None):
            print("Warning: Multiple function definitions found but no func_name specified. Will cause unintended behavior")
        
        assert func_string.count('def') > 0, print("Warning: No function definition found in func_string")
        
        self.func_name = func_string.split(":")[0].split("def")[1].split("(")[0].lstrip() if \
                         func_name is None else func_name

    
    @staticmethod
    def function2string(func_object):
        '''
        Retrieve the string description of a compiled function object
        '''
        pass

    
    def dump_bytecode(self):
        '''
        Dump the bytecode description of a compiled function
        '''
        self.compile() if self.func is None else 0==0
        return(self.func.__code__.co_code)
        
    def dump_ast(self):
        '''
        Dump the AST of a compiled function
        '''
        dedented = textwrap.dedent(self.func_string)
        ast_out = ast.parse(dedented)
        ast_out = ast.dump(ast_out, indent=4)
        return(ast_out)

    def dump_constants(self):
        '''
        Dump the constants defined in the function
        '''
        return(self.func.__code__.co_consts) 
        
    def retrieve_function(self, func_name):
        '''
        Retrieve a function from the environment using the name
        '''
        possibles = globals().copy()
        possibles.update(locals())
        if func_name not in possibles:
            print("Function not compiled properly!")
            assert False
        else:
            return(possibles.get((func_name)))


    def check_fn(self):
        '''
        Check that the function string defines a closed algorithm (i.e it has no
        free variables and does not utilize or modify global scope)
        '''
        assert len(self.func_string.__code__.co_freevars) == 0
        
        
    def compile(self):
        '''
        Compile a string description of a function using the specified compiler
        '''

        # Delete the function if it exists already
        if self.func_name in locals():
            del locals()[self.func_name]
        if self.func_name in globals():
            del globals()[self.func_name]
        
        code_obj = compile(self.func_string,optimize = self.optimization_level, filename= "<string>", mode='exec')
        exec(code_obj, globals())
        self.func = self.retrieve_function(self.func_name)
        
        if self.compiler == 'numba':
            self.func = numba.jit(cache = False, nopython = True)(self.func)
        
        return(self.func)
        
    def similarity(self, func2, inputs):
        '''
        Based on the compiler and thresholds specified, provide a continuous
        answer to how similar two functions are
        '''
        assert type(func2) == CompiledFunction, "Both functions must be CompiledFunction objects"
        assert self.compiler == func2.compiler, "Both functions have different compilers"

        comparisons = {}

        # Compare functional equivelence
        self.compile() if self.func is None else 0==0
        func2.compile() if func2.func is None else 0==0
        inps = [inputs] if not isinstance(inputs, Iterable) else inputs
        outs1 = [self.func(x) if isinstance(x,Iterable) else self.func(x) for x in inps]
        outs2 = [func2.func(x) for x in inps]

        # MSE of the difference in outputs for the same set of inputs
        #comparisons['MSE_outputs'] = sum([(outs1[i]-outs2[i])**2 for i in range(len(outs1))]) / len(outs1)

        # Levenshtein distance of their bytecode
        comparisons['LevDistance_bytecode'] = levenshtein_distance(self.dump_bytecode(), func2.dump_bytecode())

        # Levenshtein distance of their string descriptions
        comparisons['LevDistance_string'] = levenshtein_distance(self.func_string, func2.func_string)
        
        return(comparisons)
        

        
    def equals(self, func2, inputs):
        '''
        Based on the compiler and thresholds specified, provide a boolean
        answer to whether the two functions are equal
        '''

        assert type(func2) == CompiledFunction, "Both functions must be CompiledFunction objects"
        assert self.compiler == func2.compiler, "Both functions have different compilers"

        comparisons = {}

        # Compare functional equivelence
        self.compile() if self.func is None else 0==0
        func2.compile() if func2.func is None else 0==0
        
        inps = [inputs] if not isinstance(inputs, Iterable) else inputs
        #inps = [[x] if not isinstance(x, Iterable) else x for x in inps]
        outs1 = [self.func(x) if isinstance(x,Iterable) else self.func(x) for x in inps]
        outs2 = [func2.func(x) for x in inps]
        comparisons['functionally_equal'] = outs1==outs2

        # Compare the bytecode 
        comparisons['bytecode_equal'] = self.dump_bytecode() == func2.dump_bytecode()

        # Compare the constants
        comparisons['constants_equal'] = self.dump_constants() == func2.dump_constants()

        # Compare the ASTs
        comparisons['asts_equal'] = self.dump_ast() == func2.dump_ast()

        # Compare the # of local variables
        #comparisons['locals_equal'] = 

        # Compare the strings
        comparisons['encoding_equal'] = self.func_string == func2.func_string
        
        return(comparisons)

        
    def __call__(self, inp):
        '''
        Call the compiled function on an input
        '''

        self.compile() if self.func is None else 0==0
        assert self.func is not None, "Function not compiled properly"
        if isinstance(inp, Iterable):
            return(self.func(*inp))
        return(self.func(inp))


    def measure_runtime(self, inp):
        '''
        Measure the runtime of the compiled function against a specific input
        '''
        start = perf_counter()
        self.__call__(inp); 
        stop = perf_counter()
        return(stop - start)
        
    def measure_memory(self, inp):
        '''
        Measure the memory usage of the compiled function against a specified input
        '''
        ## Memory logger
        tracemalloc.clear_traces()
        tracemalloc.start()
    
        self.__call__(inp)
    
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.reset_peak()
        
        return(peak)


    def get_ops(self, exclude=['STORE_FAST', 'LOAD_CONST', 'LOAD_FAST', 'RESUME', 'RETURN_VALUE']):
        '''
        Get a list of primitive operations for the function
        using the bytecode
        '''
        return([x.opname+"_"+x.argrepr for x in dis.get_instructions(self.func) if x.opname not in exclude])

    def describe_ops(self, exclude=['STORE_FAST', 'LOAD_CONST', 'LOAD_FAST', 'RESUME', 'RETURN_VALUE']):
        '''
        Return the primitive operations and their counts
        '''
        return(Counter(self.get_ops(exclude=exclude)))


    def encode_ops(self, ops_to_encode=['COMPARE_OP_==', 'COMPARE_OP_>', 'COMPARE_OP_!=', 'COMPARE_OP_<', 'COMPARE_OP_<=', 'COMPARE_OP_>=',
                                        'BINARY_OP_+','BINARY_OP_*', 'BINARY_OP_-', 'BINARY_OP_**', 'BINARY_OP_+=', 'BINARY_OP_//', 'FOR_ITER_to'], notify=False):
        
        op_descriptions = self.describe_ops()
        out_vector = []
        
        for op in ops_to_encode: # For the operations in the CompiledFunction
            
            if op in op_descriptions.keys(): # If the op is specified in ops_to_encode
                out_vector.append(op_descriptions[op])

            else:
                out_vector.append(0)
                if notify:
                    print("The following operations was found but not specified in the encoding: ", op)
        
        return(out_vector)
        
    def count_ops(self, context = {}):
        '''
        Estimate the number of distinct operations performed
        by a function using the bytecode
        '''
        processed: Set[CodeType] = set()
        stack: List[CodeType] = [self.func.__code__]
    
        ops = []
        while stack:
            codeobj = stack.pop()
            if codeobj in processed: continue
    
            processed.add(codeobj)
            ops.extend(get_ops_(dis.Bytecode(codeobj)))
    
            # handle local scoped functions
            for const in codeobj.co_consts:
                if isinstance(const, CodeType):
                    stack.append(const)
    
            for name in get_fn_names_(codeobj, context):
                if name not in processed:
                    try:
                        func = context.get(name)
                        stack.append(func.__code__)
                    except KeyError:
                        print(f"Warning: Function '{name}' not found in globals")
        return(len(ops))





################################################################ O Term ###############################################################################
# helpers to compute o_term
builtin_names = set(__builtins__.keys())
ops_to_match = ['BINARY_OP', 'COMPARE_OP']

def get_fn_names_(codeobj: CodeType, context):
    names = []
    for name in codeobj.co_names:
        if name in context and name not in builtin_names:
            if isinstance(context.get(name), FunctionType):
                names.append(name)
    return names

def get_ops_(bytecode: dis.Bytecode):
    ops = [instr.opname for instr in bytecode if instr.opname in ops_to_match]
    return ops

# must pass execution context of |func|
def o_term(func, context):
    '''
    Estimate the number of distinct operations performed
    by a function using the bytecode
    '''
    processed: Set[CodeType] = set()
    stack: List[CodeType] = [func.__code__]

    ops = []
    while stack:
        codeobj = stack.pop()
        if codeobj in processed: continue

        processed.add(codeobj)
        ops.extend(get_ops_(dis.Bytecode(codeobj)))

        # handle local scoped functions
        for const in codeobj.co_consts:
            if isinstance(const, CodeType):
                stack.append(const)

        for name in get_fn_names_(codeobj, context):
            if name not in processed:
                try:
                    func = context.get(name)
                    stack.append(func.__code__)
                except KeyError:
                    print(f"Warning: Function '{name}' not found in globals")
    return(len(ops))


################################################################ M Term ###############################################################################
def m_term(func, inp, verbose = 1):
    '''
    Estimate the runtime memory usage of a function for a single input
    '''
        
    ## Memory logger
    tracemalloc.clear_traces()
    tracemalloc.start()

    try:
        func(inp); 
    except Exception as e:
        print("Could not run function. \n ") if verbose else ""
        print(e) if verbose else ""
        tracemalloc.stop()
        tracemalloc.reset_peak()
        return(None)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.reset_peak()
    
    return(peak)




def m_term_multisize(func, inps, sizes, reduction = lambda x: np.median(x), verbose = 1):
    '''
    Estimate the runtime memory scaling of a function for multiple inputs of 
    varying sizes. This is done by calling m_term on each individual output, then fitting
    a line to the size vs. m_term relationship, and outputting the slope. 
    '''
    memory_uses = {}
    for i in range(len(inps)):
        inp = inps[i]
        inpsize = sizes[i]
        mem = m_term(func, inp, verbose = verbose)
        if mem is not None:
            if inpsize in memory_uses:
                memory_uses[inpsize].append(mem)
            else:
                memory_uses[inpsize] = [mem]
    memory_avgd = [(k,reduction(v)) for k,v in zip(memory_uses.keys(), memory_uses.values()) if len(v) > 0]
    
    # Extract x and y values from the tuples
    x = np.array([t[0] for t in memory_avgd if (t[0] is not None) and (t[1] is not None)])
    y = np.array([t[1] for t in memory_avgd if (t[0] is not None) and (t[1] is not None)])
    
    # Fit a line with zero intercept
    x = x[:,np.newaxis]
    slope, _, _, _ = np.linalg.lstsq(x, y)
    
    return(slope, x, y)
    
################################################################ C Term ###############################################################################
def c_term(func, inp, use_time = True, verbose = 1):
    '''
    Estimate the runtime computations performed by a function for a single input.
    Since it is not possible to get compiler level operations for a user-defined function,
    we instead calculate a proxy value based on the number of times any builtin method is called.
    '''
    
    if use_time: # Use clocktime
        try:
            start = perf_counter()
            func(inp); 
            stop = perf_counter()
        except Exception as e:
            print("Could not run function. \n ") if verbose else ""
            print(e) if verbose else ""
            return(None)
        return(stop - start)
    else:
        raise NotImplementedError("c-term only supports clocktime at the moment")



def c_term_multisize(func, inps, sizes, reduction = lambda x: np.median(x), verbose = 1):
    '''
    Estimate the runtime compute scaling of a function for multiple inputs of 
    varying sizes. This is done by calling c_term on each individual output, then fitting
    a line to the size vs. c_term relationship, and outputting the slope. 
    '''
    runtimes = {}
    for i in range(len(inps)):
        inp = inps[i]
        inpsize = sizes[i]
        runtime = c_term(func, inp, verbose = verbose)
        if runtime is not None:
            if inpsize in runtimes:
                runtimes[inpsize].append(runtime)
            else:
                runtimes[inpsize] = [runtime]
    runtimes_avgd = [(k,reduction(v)) for k,v in zip(runtimes.keys(), runtimes.values()) if len(v) > 0]
    
    # Extract x and y values from the tuples
    x = np.array([t[0] for t in runtimes_avgd if (t[0] is not None) and (t[1] is not None)])
    y = np.array([t[1] for t in runtimes_avgd if (t[0] is not None) and (t[1] is not None)])
    
    # Fit a line with zero intercept
    #slope = np.polyfit(x, y, 1, full=False)[0]
    x = x[:,np.newaxis]
    slope, _, _, _ = np.linalg.lstsq(x, y)
    
    return(slope, x, y)


################################################################ MISC ###############################################################################

def get_func_from_src(src, function_name):
    src = src.replace("print(", "pass#")
    exec(src)
    possibles = globals().copy()
    possibles.update(locals())
    return(possibles.get((function_name)))

def n_lists_of_size_s(n = 10, s = 5):
    return([list(np.random.randint(low = 0, high = 10000, size = s)) for _ in range(n)])
    
def describe_function(func, domain):
    '''
    Calculate the EMOC score for a single algorithm,
    provided as an ananomyous or named pure function
    
    E = Evaluation
    M = Memory
    O = Operations
    C = Complexity

    Args:
        func (callable): The function to be analyzed, func(domain[i]) must be defined for all i
        domain (iterable or Domain object): Input iterable to be fed into function to analyze runtime behavior
    '''
    # Calculate E component - Evaluation accuracy
    e = 1
    
    # Calculate O component - IoU operation overlap
    o = o_term(func)
    

    # Runtime analysis
    data = {}


    ## For every input...
    for d in domain:

        ## Get the size of the input
        inp_size = len(d)

        inp = deepcopy(d) # Prevent overwriting
        
        ## Calculate the m-term
        m_ = m_term(func, inp)

        ## Calculate the c-term
        c_ = c_term(func, inp)

        ## Organize these by size

        ## If we've seen this size of input before
        if inp_size in data.keys():
            
            old_vals = data[inp_size]

            # Overwrite values if new sample is worst-case so far
            if m_ is not None and m_ > old_vals['memory']:
                old_vals['memory'] = m_

            if c_ is not None and c_ > old_vals['computations']:
                old_vals['computations']

            data[inp_size] = old_vals

        ## If we haven't seen an input of this size...
        elif inp_size not in data.keys():
            m_ = m_ if m_ is not None else -1
            c_ = c_ if c_ is not None else -1
            if max([m_, c_]) > 0:
                newdict = {'memory': m_, 'computations': c_}
                data[inp_size] = newdict


    # Perform reduction of the runtime data
    sizes = list(data.keys())
    memories = [data[input_size]['memory'] for input_size in sizes]
    computations = [data[input_size]['computations'] for input_size in sizes]
    
    ## For the memory, take the total squared deviation from O(1) memory usage 
    m = np.max([(y - x) for x,y in zip(sizes, memories)])

    ## For the computation, take the total squared deviation from O(1) complexity
    c = np.max([(y - x) for x,y in zip(sizes, computations)])

    return(e,m,o,c)






def reindex_clusters(labels, cluster_labels):
    """Reindex cluster labels to match the order of true labels."""

    # Create a mapping from cluster labels to true labels
    mapping = {}
    for i in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == i)[0]
        true_labels_in_cluster = labels[cluster_indices]
        if len(true_labels_in_cluster) == 0:
            print("Unique cluster labels: ", np.unique(cluster_labels))
            print("Unique dataset labels: ", np.unique(labels))
            continue
        most_common_true_label = np.bincount(true_labels_in_cluster).argmax()
        mapping[i] = most_common_true_label

    # Reindex the cluster labels using the mapping
    reindexed_cluster_labels = np.array([mapping[label] for label in cluster_labels])

    return reindexed_cluster_labels
    

def compare_functions(func1, func2, domain = {-1, -2, 1,2,3}):
    '''
    Calculate the EMOC score for two python algorithms,
    provided as ananomyous or named pure functions
    
    E = Evaluation
    M = Memory
    O = Operations
    C = Complexity

    Args:
        func1 (callable): The first function to be compared, func1(domain[i]) must be defined for all i
        func2 (callable): The second function to be compared, func2(domain[i]) must be defined for all i
        domain (iterable): Input values to be fed into both function to analyze runtime behavior
    '''
    # TODO - pre-process input functions
    func1 = func1.replace("print(", "#print(")
    func2 = func2.replace("print(", "#print(")
    
    # Calculate O component - IoU operation overlap
    o_term = 0
    bc1 = dis.Bytecode(func1)
    bc2 = dis.Bytecode(func2)
    # ideally, we should weight ops differently
    ops_to_match = ['BINARY_OP', 'COMPARE_OP']
    ops1 = [instr.opname for instr in bc1 if instr.opname in ops_to_match]
    ops2 = [instr.opname for instr in bc2 if instr.opname in ops_to_match]
    ops1 = Counter(ops1)
    ops2 = Counter(ops2)
    allops = list(ops1.keys()) + list(ops2.keys())
    for key in allops:
        if key in ops1 and key in ops2:
            o_term += np.abs(ops1[key] - ops2[key])
        elif key in ops1:
            o_term += ops1[key]
        elif key in ops2:
            o_term += ops2[key]

    # Calculate the E component - Evaluation similarity
    e_term = 0
    for d in domain:
        f1_result = func1(d)
        f2_result = func2(d)
        if isinstance(d, list):
            # compute median of list
            f1_result = sum(f1_result) / len(f1_result)
            f2_result = sum(f2_result) / len(f2_result)
        diff =  np.abs(f1_result - f2_result) / np.max([f1_result, f2_result])
        e_term += np.abs(diff)

    # Calculate the M component - Memory similarity
    m_term = 0
    for d in domain:
        tracemalloc.start()
        func1(d)
        current, peak = tracemalloc.get_traced_memory()
        f1_result = peak
        tracemalloc.stop()

        tracemalloc.start()
        func2(d)
        current, peak = tracemalloc.get_traced_memory()
        f2_result = peak
        tracemalloc.stop()
        diff =  np.abs(f1_result - f2_result) / np.max([f1_result+0.1, f2_result+0.1])
        m_term += diff

    # Calculate the C component - Complexity
    # TODO: NOT YET IMPLEMENTED
    
    return(o_term + e_term + m_term)