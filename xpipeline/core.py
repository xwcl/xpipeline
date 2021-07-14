import os
import psutil
import numpy
import numba
import logging

log = logging.getLogger(__name__)

class YellingProxy:
    def __init__(self, package):
        self.package = package

    def __getattr__(self, name: str):
        raise AttributeError(
            f"Package {self.package} is not installed (or failed to import)"
        )

try:
    import mkl
    HAVE_MKL = True
    mkl_set_num_threads = mkl.set_num_threads
except ImportError:
    HAVE_MKL = False
    mkl = YellingProxy('mkl-service')
    def mkl_set_num_threads(n):
        log.debug('Ignoring call to mkl_set_num_threads')



def determine_max_threads():
    count = os.cpu_count()
    if 'OMP_NUM_THREADS' in os.environ:
        count = min(count, int(os.environ['OMP_NUM_THREADS']))    
    numba_count, mkl_count = count, count
    if 'NUMBA_NUM_THREADS' in os.environ:
        numba_count = min(numba_count, int(os.environ['NUMBA_NUM_THREADS']))
    if 'MKL_NUM_THREADS' in os.environ:
        mkl_count = min(mkl_count, int(os.environ['MKL_NUM_THREADS']))
    return numba_count, mkl_count

NUMBA_MAX_THREADS, MKL_MAX_THREADS = determine_max_threads()

def set_num_threads(n_threads, n_mkl_threads=None):
    if HAVE_MKL:
        if n_mkl_threads > MKL_MAX_THREADS:
            log.debug(f'{n_mkl_threads=} was > {MKL_MAX_THREADS=}')
            n_mkl_threads = MKL_MAX_THREADS
        log.debug(f'Setting {n_mkl_threads=}')
        mkl_set_num_threads(n_mkl_threads)
    else:
        if n_mkl_threads is not None:
            log.debug(f'No MKL service, adding {n_mkl_threads=} to total')
            n_threads = n_threads + n_mkl_threads
    if n_threads > NUMBA_MAX_THREADS:
        log.debug(f'{n_threads=} was > {NUMBA_MAX_THREADS=}')
        n_threads = NUMBA_MAX_THREADS
    numba.set_num_threads(n_threads)
    log.debug(f'Setting {n_threads=}')

def get_array_module(arr):
    """Returns `dask.array` if `arr` is a `dask.array.core.Array`, or
    numpy if `arr` is a NumPy ndarray.

    Use to write code that can handle both, e.g.::

        xp = get_array_module(input_array)
        xp.sum(input_array)
    """
    if isinstance(arr, numpy.ndarray):
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

    def __init__(self, inputs):
        self.items = inputs

    def _wrap_callable(self, callable, _delayed_kwargs):
        raise NotImplementedError(
            "Use a subclass like LazyPipelineCollection or EagerPipelineCollection"
        )

    def zip_map(self, callable, *args, _delayed_kwargs=None, **kwargs):
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
        for idx, x in enumerate(self.items):
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
            out.append(
                self._wrap_callable(callable, _delayed_kwargs)(
                    x, *new_args, **new_kwargs
                )
            )
        return self.__class__(out)

    def map(self, callable, *args, _delayed_kwargs=None, **kwargs):
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
        return self.__class__(
            [
                self._wrap_callable(callable, _delayed_kwargs)(x, *args, **kwargs)
                for x in self.items
            ]
        )

    def collect(self, callable, *args, _delayed_kwargs=None, **kwargs):
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
        return self._wrap_callable(callable, _delayed_kwargs)(
            self.items, *args, **kwargs
        )

    def compute(self):
        """Pass `self.inputs` to `dask.compute` and return the result
        of executing the pipeline
        """
        return dask.compute(self.items)[0]

    def persist(self):
        """Pass `self.inputs` to `dask.persist` and return a delayed
        reference to the result of executing the pipeline
        """
        return dask.persist(self.items)[0]

    def end(self):
        """Return the Delayed instances for the pipeline outputs"""
        return self.items

    def with_new_contents(self, collection):
        return self.__class__(collection)


class LazyPipelineCollection(PipelineCollection):
    def _wrap_callable(self, callable, _delayed_kwargs):
        kwargs = _delayed_kwargs if _delayed_kwargs is not None else {}
        return dask.delayed(callable, **kwargs)


class EagerPipelineCollection(LazyPipelineCollection):
    def _wrap_callable(self, callable, _delayed_kwargs):
        return callable

    def compute(self):
        return self.items

def _dask_reduce_bitwise_or(chunks):
    if hasattr(chunks, 'dtype'):
        dtype = chunks.dtype
    else:
        dtype = chunks[0].dtype
    if hasattr(chunks, 'shape'):
        shape = chunks.shape[1:]
    else:
        shape = chunks[0].shape[1:]
    out = numpy.zeros(shape, dtype)
    for chunk in chunks:
        out |= numpy.bitwise_or.reduce(chunk, axis=0)
    return out
def reduce_bitwise_or(arr):
    xp = get_array_module(arr)
    if xp is dask_array:
        return dask_array.blockwise(_dask_reduce_bitwise_or, 'jk', arr, 'ijk')
    else:
        return numpy.bitwise_or.reduce(arr, axis=0)
