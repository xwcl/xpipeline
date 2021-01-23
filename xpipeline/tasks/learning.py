import logging
import random
import numpy as np

import dask

log = logging.getLogger(__name__)

@dask.delayed(nout=2)
def train_test_split(array, test_frac, random_state=0):
    random.seed(random_state)
    log.debug(f'{array=}')
    n_elems = array.shape[0]
    elements = range(n_elems)
    mask = np.zeros(n_elems, dtype=bool)
    test_count = int(test_frac * n_elems)
    test_elems = random.choices(elements, k=test_count)
    for idx in test_elems:
        mask[idx] = True
    train_subarr, test_subarr = array[~mask], array[mask]
    log.info(f"Cross-validation reserved {100 * test_frac:2.1f}% of inputs")
    log.info(f'Split {n_elems} into {train_subarr.shape[0]} and {test_subarr.shape[0]}')
    return train_subarr, test_subarr
