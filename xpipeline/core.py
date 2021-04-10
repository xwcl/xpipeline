import warnings
import numpy
import dask
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    warnings.warn('PyTorch is unavailable, attempts to use PyTorch features will error')
    torch = None
    HAVE_TORCH = False
try:
    import cupy
    HAVE_CUPY = True
except ImportError:
    warnings.warn('CuPy is unavailable, attempts to use GPU features will error')
    cupy = None
    HAVE_CUPY = False
try:
    import dask.array as dask_array
    from dask.array import core as dask_array_core
    HAVE_DASK = True
except ImportError:
    warnings.warn('Dask is unavailable, attempts to use distributed features will error')
    dask_array = None
    dask_array_core = None
    HAVE_DASK = False



def get_array_module(arr):
    '''Returns `dask.array` if `arr` is a `dask.array.core.Array`, or
    numpy if `arr` is a NumPy ndarray.

    Use to write code that can handle both, e.g.::

        xp = get_array_module(input_array)
        xp.sum(input_array)
    '''
    if HAVE_DASK and isinstance(arr, dask_array_core.Array):
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


class PipelineCollection:
    """Construct sequences of delayed operations on a collection of
    inputs with a chainable API to map callables to inputs
    """

    def __init__(self, d_inputs):
        self.d_inputs = d_inputs

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
            as-is. (Iterables that are not the same length as `d_inputs`
            are currently unsupported.)

        Returns
        -------
        coll : PipelineCollection
            New PipelineCollection with results for chaining
        """
        out = []
        for idx, x in enumerate(self.d_inputs):
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
            out.append(callable(x, *new_args, **new_kwargs))
        return PipelineCollection(out)

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
        coll : PipelineCollection
            New PipelineCollection with results for chaining
        """
        return PipelineCollection([callable(x, *args, **kwargs) for x in self.d_inputs])

    def collect(self, callable, *args, **kwargs):
        """
        Apply function to entire collection

        Parameters
        ----------
        callable : callable
            Function accepting list of previous stage outputs as input.
            Will be passed *args and **kwargs
        *args, **kwargs
            Arguments passed through to `callable`
        """
        return callable(self.d_inputs, *args, **kwargs)

    def compute(self):
        '''Shortcut for ``PipelineCollection.collect(dask.compute)``
        '''
        return dask.compute(self.d_inputs)

    def end(self):
        return self.d_inputs
