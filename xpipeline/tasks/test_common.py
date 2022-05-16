from importlib import resources
import numpy as np

import pytest

def get_naco_betapic_data():
    res_handle = resources.open_binary(
        "xpipeline.ref", "naco_betapic_preproc_absil2013_gonzalez2017.npz"
    )
    data = np.load(res_handle)
    return data

@pytest.fixture
def naco_betapic_data():
    return get_naco_betapic_data()