"""
Vendored from scipy @ c770843f37f807024652ad425b82271d8b0eb438
on Nov 10 2021. Check for updates at:
https://github.com/scipy/scipy/blob/master/scipy/sparse/linalg/isolve/tests/test_lsqr.py

Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import scipy.sparse
import scipy.sparse.linalg
from .lsqr import lsqr
from time import time

from .core import cupy, HAVE_CUPY

n = 35


def make_G():
    # Set up a test problem
    G = np.eye(n)
    normal = np.random.normal
    norm = np.linalg.norm

    for jj in range(5):
        gg = normal(size=n)
        hh = gg * gg.T
        G += (hh + hh.T) * 0.5
        G += normal(size=n) * normal(size=n)
    return G

G = pytest.fixture(make_G)

def make_b():
    return np.random.normal(size=n)
b = pytest.fixture(make_b)

tol = 1e-10
show = False
maxit = None

@pytest.mark.parametrize('xp', [
    pytest.param(cupy, marks=pytest.mark.skipif(not HAVE_CUPY, reason="No GPU support")),
    np
])
def test_basic(xp, G, b):
    # compute with normal dense solver
    svx = np.linalg.solve(G, b)
    # move to GPU if needed
    G = xp.asarray(G)
    b = xp.asarray(b)
    b_copy = b.copy()
    xo, *_ = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
    if xp is cupy:
        assert_array_equal(b_copy.get(), b.get())
    else:
        assert_array_equal(b_copy, b)

    # compare to dense solver
    if xp is cupy:
        xo = xo.get()
    assert_allclose(xo, svx, atol=tol, rtol=tol)

    # Now the same but with damp > 0.
    # This is equivalent to solving the extented system:
    # ( G      ) @ x = ( b )
    # ( damp*I )       ( 0 )
    damp = 1.5
    xo, *_ = lsqr(
        G, b, damp=damp, show=show, atol=tol, btol=tol, iter_lim=maxit)

    Gext = xp.r_[G, damp * np.eye(G.shape[1])]
    bext = xp.r_[b, np.zeros(G.shape[1])]
    svx, *_ = xp.linalg.lstsq(Gext, bext, rcond=None)
    if xp is cupy:
        xo = xo.get()
        svx = svx.get()
    assert_allclose(xo, svx, atol=tol, rtol=tol)


def test_gh_2466():
    row = np.array([0, 0])
    col = np.array([0, 1])
    val = np.array([1, -1])
    A = scipy.sparse.coo_matrix((val, (row, col)), shape=(1, 2))
    b = np.asarray([4])
    lsqr(A, b)


@pytest.mark.parametrize('xp', [
    pytest.param(cupy, marks=pytest.mark.skipif(not HAVE_CUPY, reason="No GPU support")),
    np
])
def test_well_conditioned_problems(xp):
    # Test that sparse the lsqr solver returns the right solution
    # on various problems with different random seeds.
    # This is a non-regression test for a potential ZeroDivisionError
    # raised when computing the `test2` & `test3` convergence conditions.
    n = 10
    if HAVE_CUPY and xp is cupy:
        import cupyx.scipy.sparse
        sparse = cupyx.scipy.sparse
    else:
        sparse = scipy.sparse
    A_sparse = sparse.eye(n, n)
    A_dense = A_sparse.toarray()

    with np.errstate(invalid='raise'):
        for seed in range(30):
            rng = np.random.RandomState(seed + 10)
            beta = xp.asarray(rng.rand(n))
            beta[beta == 0] = 0.00001  # ensure that all the betas are not null
            b = A_sparse @ beta[:, np.newaxis]
            output = lsqr(A_sparse, b, show=show)

            # Check that the termination condition corresponds to an approximate
            # solution to Ax = b
            assert_equal(output[1], 1)
            solution = output[0]

            # Check that we recover the ground truth solution
            if xp is cupy:
                assert_allclose(solution.get(), beta.get())
            else:
                assert_allclose(solution, beta)

            # Sanity check: compare to the dense array solver
            reference_solution = xp.linalg.solve(A_dense, b).ravel()
            if xp is cupy:
                assert_allclose(solution.get(), reference_solution.get())
            else:
                assert_allclose(solution, reference_solution)

@pytest.mark.parametrize('xp', [
    pytest.param(cupy, marks=pytest.mark.skipif(not HAVE_CUPY, reason="No GPU support")),
    np
])
def test_b_shapes(xp):
    # Test b being a scalar.
    A = xp.array([[1.0, 2.0]])
    b = 3.0
    x = lsqr(A, b)[0]
    diffnorm = xp.linalg.norm(A.dot(x) - b)
    if xp is cupy:
        diffnorm = diffnorm.get()
    assert diffnorm == pytest.approx(0)

    # Test b being a column vector.
    A = np.eye(10)
    b = np.ones((10, 1))
    x = lsqr(A, b)[0]
    diffnorm = xp.linalg.norm(A.dot(x) - b.ravel())
    if xp is cupy:
        diffnorm = diffnorm.get()
    assert diffnorm == pytest.approx(0)

# @pytest.mark.parametrize('xp', [
#     pytest.param(cupy, marks=pytest.mark.skipif(not HAVE_CUPY, reason="No GPU support")),
#     np
# ])
def test_initialization(G, b):
    # Test the default setting is the same as zeros
    b_copy = b.copy()
    x_ref = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
    x0 = np.zeros(x_ref[0].shape)
    x = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
    assert_array_equal(b_copy, b)
    assert_allclose(x_ref[0], x[0])

    # Test warm-start with single iteration
    x0 = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=1)[0]
    x = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
    assert_allclose(x_ref[0], x[0])
    assert_array_equal(b_copy, b)


if __name__ == "__main__":
    G, b = make_G(), make_b()
    svx = np.linalg.solve(G, b)

    tic = time()
    X = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
    xo = X[0]
    phio = X[3]
    psio = X[7]
    k = X[2]
    chio = X[8]
    mg = np.amax(G - G.T)
    if mg > 1e-14:
        sym = 'No'
    else:
        sym = 'Yes'

    print('LSQR')
    print("Is linear operator symmetric? " + sym)
    print("n: %3g  iterations:   %3g" % (n, k))
    print("Norms computed in %.2fs by LSQR" % (time() - tic))
    print(" ||x||  %9.4e  ||r|| %9.4e  ||Ar||  %9.4e " % (chio, phio, psio))
    print("Residual norms computed directly:")
    print(" ||x||  %9.4e  ||r|| %9.4e  ||Ar||  %9.4e" % (norm(xo),
                                                         norm(G*xo - b),
                                                         norm(G.T*(G*xo-b))))
    print("Direct solution norms:")
    print(" ||x||  %9.4e  ||r|| %9.4e " % (norm(svx), norm(G*svx - b)))
    print("")
    print(" || x_{direct} - x_{LSQR}|| %9.4e " % norm(svx-xo))
    print("")
