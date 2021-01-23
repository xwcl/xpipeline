import numpy as np
from .learning import train_test_split

def test_train_test_split():
    n_total = 10
    data = np.arange(n_total)
    train_a, test_a = train_test_split(data, 0.5, random_state=0).compute()
    assert train_a.size + test_a.size == data.size, "Lost elements in split"
    assert train_a.shape[0] == n_total // 2, "Wrong split"
    assert test_a.shape[0] == n_total // 2, "Wrong split"
    repeat_train_a, repeat_test_a = train_test_split(data, 0.5, random_state=0).compute()
    assert np.all(train_a == repeat_train_a), "Repeat call with same RNG state not reproducible"
    assert np.all(test_a == repeat_test_a), "Repeat call with same RNG state not reproducible"
    train_b, test_b = train_test_split(data, 0.5, random_state=1).compute()
    assert np.any(train_a != train_b), "Repeat call with new RNG state not different"
    assert np.any(test_a != test_b), "Repeat call with new RNG state not different"
