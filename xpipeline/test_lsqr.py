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
from scipy.sparse.linalg import lsqr
from time import time

# Set up a test problem
n = 35
G = np.eye(n)
normal = np.random.normal
norm = np.linalg.norm

for jj in range(5):
    gg = normal(size=n)
    hh = gg * gg.T
    G += (hh + hh.T) * 0.5
    G += normal(size=n) * normal(size=n)

b = normal(size=n)

tol = 1e-10
show = False
maxit = None


def test_basic():
    b_copy = b.copy()
    xo, *_ = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
    assert_array_equal(b_copy, b)

    svx = np.linalg.solve(G, b)
    assert_allclose(xo, svx, atol=tol, rtol=tol)

    # Now the same but with damp > 0.
    # This is equivalent to solving the extented system:
    # ( G      ) @ x = ( b )
    # ( damp*I )       ( 0 )
    damp = 1.5
    xo, *_ = lsqr(
        G, b, damp=damp, show=show, atol=tol, btol=tol, iter_lim=maxit)

    Gext = np.r_[G, damp * np.eye(G.shape[1])]
    bext = np.r_[b, np.zeros(G.shape[1])]
    svx, *_ = np.linalg.lstsq(Gext, bext, rcond=None)
    assert_allclose(xo, svx, atol=tol, rtol=tol)


def test_gh_2466():
    row = np.array([0, 0])
    col = np.array([0, 1])
    val = np.array([1, -1])
    A = scipy.sparse.coo_matrix((val, (row, col)), shape=(1, 2))
    b = np.asarray([4])
    lsqr(A, b)


def test_well_conditioned_problems():
    # Test that sparse the lsqr solver returns the right solution
    # on various problems with different random seeds.
    # This is a non-regression test for a potential ZeroDivisionError
    # raised when computing the `test2` & `test3` convergence conditions.
    n = 10
    A_sparse = scipy.sparse.eye(n, n)
    A_dense = A_sparse.toarray()

    with np.errstate(invalid='raise'):
        for seed in range(30):
            rng = np.random.RandomState(seed + 10)
            beta = rng.rand(n)
            beta[beta == 0] = 0.00001  # ensure that all the betas are not null
            b = A_sparse @ beta[:, np.newaxis]
            output = lsqr(A_sparse, b, show=show)

            # Check that the termination condition corresponds to an approximate
            # solution to Ax = b
            assert_equal(output[1], 1)
            solution = output[0]

            # Check that we recover the ground truth solution
            assert_allclose(solution, beta)

            # Sanity check: compare to the dense array solver
            reference_solution = np.linalg.solve(A_dense, b).ravel()
            assert_allclose(solution, reference_solution)


def test_b_shapes():
    # Test b being a scalar.
    A = np.array([[1.0, 2.0]])
    b = 3.0
    x = lsqr(A, b)[0]
    assert norm(A.dot(x) - b) == pytest.approx(0)

    # Test b being a column vector.
    A = np.eye(10)
    b = np.ones((10, 1))
    x = lsqr(A, b)[0]
    assert norm(A.dot(x) - b.ravel()) == pytest.approx(0)


def test_initialization():
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
