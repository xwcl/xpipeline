import numpy
import dask

class YellingProxy:
    def __init__(self, package):
        self.package = package
    def __getattr__(self, name: str):
        raise AttributeError(f"Package {self.package} is not installed (or failed to import)")

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    torch = YellingProxy("PyTorch")
    HAVE_TORCH = False
try:
    import cupy
    HAVE_CUPY = True
except ImportError:
    cupy = YellingProxy("CuPy")
    HAVE_CUPY = False

import dask.array as dask_array
from dask.array import core as dask_array_core

newaxis = numpy.newaxis

def get_array_module(arr):
    '''Returns `dask.array` if `arr` is a `dask.array.core.Array`, or
    numpy if `arr` is a NumPy ndarray.

    Use to write code that can handle both, e.g.::

        xp = get_array_module(input_array)
        xp.sum(input_array)
    '''
    if isinstance(arr, dask_array_core.Array):
        return dask_array
    elif HAVE_CUPY and isinstance(arr, cupy.ndarray):
        return cupy
    elif isinstance(arr, numpy.ndarray):
        return numpy
    else:
        raise ValueError("Unrecognized type passed to get_array_module")


def _is_iterable_arg(obj):
    if isinstance(obj, str):
        # technically iterable, but distributing individual characters
        # is never what we want
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class LazyPipelineCollection:
    """Construct sequences of delayed operations on a collection of
    inputs with a chainable API to map callables to inputs
    """

    def __init__(self, inputs):
        self.collection = inputs

    def _wrap_callable(self, callable):
        return dask.delayed(callable)

    def zip_map(self, callable, *args, **kwargs):
        """Apply function to inputs with varying argument values
        selected from iterable arguments or keyword arguments

        Parameters
        ----------
        callable : callable
            Function accepting one entry of the set of inputs
            as its first argument. Will be passed *args and **kwargs
        *args, **kwargs
            If an argument is iterable, its i'th entry will
            be supplied as that argument when calling `callable`
            on the i'th input. Non-iterable arguments are passed
            as-is. (Iterables that are not the same length as `inputs`
            are currently unsupported.)

        Returns
        -------
        coll : LazyPipelineCollection
            New LazyPipelineCollection with results for chaining
        """
        out = []
        for idx, x in enumerate(self.collection):
            new_args = []
            new_kwargs = {}
            for arg in args:
                if _is_iterable_arg(arg):
                    new_args.append(arg[idx])
                else:
                    new_args.append(arg)
            for kw, arg in kwargs.items():
                if _is_iterable_arg(arg):
                    new_kwargs[kw] = arg[idx]
                else:
                    new_kwargs[kw] = arg
            out.append(self._wrap_callable(callable)(x, *new_args, **new_kwargs))
        return self.__class__(out)

    def map(self, callable, *args, **kwargs):
        """Apply function individually to all inputs

        Parameters
        ----------
        callable : callable
            Function accepting one entry of the set of inputs
            as its first argument. Will be passed *args and **kwargs
        *args, **kwargs
            Arguments passed through to `callable`

        Returns
        -------
        coll : LazyPipelineCollection
            New LazyPipelineCollection with results for chaining
        """
        return self.__class__([self._wrap_callable(callable)(x, *args, **kwargs) for x in self.collection])

    def collect(self, callable, *args, **kwargs):
        """
        Apply function to entire collection, returning Delayed

        Parameters
        ----------
        callable : callable
            Function accepting list of previous stage outputs as input.
            Will be passed *args and **kwargs
        *args, **kwargs
            Arguments passed through to `callable`
        """
        return self._wrap_callable(callable)(self.collection, *args, **kwargs)

    def compute(self):
        '''Pass `self.inputs` to `dask.compute` and return the result
        of executing the pipeline
        '''
        return dask.compute(self.collection)

    def end(self):
        '''Return the Delayed instances for the pipeline outputs
        '''
        return self.collection

class EagerPipelineCollection(LazyPipelineCollection):
    def _wrap_callable(self, callable):
        return callable
    def compute(self):
        raise NotImplementedError("EagerPipelineCollection computes as it goes; access instance.collection for contents")
