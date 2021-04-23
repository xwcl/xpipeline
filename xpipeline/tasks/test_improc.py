import numpy as np
import pytest
import dask.array as da

from .improc import (
    rough_peak_in_box, unwrap_image, wrap_vector, cartesian_coords,
    f_test, shift2    
)


def test_rough_peak():
    peak_x, peak_y = 6, 5
    guess_x, guess_y = 6, 6
    box_x, box_y = 4, 4
    data = np.zeros((10, 10))
    _, found = rough_peak_in_box(data, (guess_y, guess_x), (box_y, box_x))
    assert not found
    data[peak_y, peak_x] = 1
    (loc_y, loc_x), found = rough_peak_in_box(data, (guess_y, guess_x), (box_y, box_x))
    assert found
    assert loc_x == peak_x
    assert loc_y == peak_y
    # the peak in the box should still be the peak even if
    # it's not the global peak
    data[0, 0] = 10
    (loc_y, loc_x), found = rough_peak_in_box(data, (guess_y, guess_x), (box_y, box_x))
    assert found
    assert loc_x == peak_x
    assert loc_y == peak_y


@pytest.mark.parametrize('xp', [np, da])
def test_wrap_2d(xp):
    image = xp.arange(9).reshape(3, 3)
    mask = np.ones(image.shape, dtype=bool)
    vec, subset_idxs = unwrap_image(image, mask)
    assert np.all(image == wrap_vector(vec, image.shape, subset_idxs))

@pytest.mark.parametrize('xp', [np, da])
def test_wrap_3d(xp):
    imcube = xp.arange(27).reshape(3, 3, 3)
    mask = np.ones(imcube.shape, dtype=bool)
    vec, subset_idxs = unwrap_image(imcube, mask)
    assert np.all(imcube == wrap_vector(vec, imcube.shape, subset_idxs))

@pytest.mark.parametrize('xp', [np, da])
def test_unwrap_2d(xp):
    image = np.zeros((3, 3))
    mask = np.ones((3, 3), dtype=bool)
    image[1, 1] = 1
    if xp is da:
        image = da.from_array(image)
    mask[1, 1] = False
    mtx, subset_idxs = unwrap_image(image, mask)
    assert mtx.shape[0] == 8
    assert subset_idxs[1].shape[0] == 8, "Not the right number of X indices"
    assert subset_idxs[0].shape[0] == 8, "Not the right number of Y indices"
    assert np.max(mtx) == 0

@pytest.mark.parametrize('xp', [np, da])
def test_unwrap_3d(xp):
    imcube = np.zeros((3, 3, 3))
    mask = np.ones(imcube.shape, dtype=bool)

    # set one nonzero pixel, and mask it out so we can tell if
    # masking worked when it's not present in the output
    imcube[1, 1, 1] = 1
    mask[1, 1, 1] = False
    if xp is da:
        imcube = da.from_array(imcube)
    mtx, subset_idxs = unwrap_image(imcube, mask)
    nonzero_pix = 3 * 3 * 3 - 1
    assert mtx.shape[0] == nonzero_pix
    assert subset_idxs[0].shape[0] == nonzero_pix, "Not the right number of Z indices"
    assert subset_idxs[1].shape[0] == nonzero_pix, "Not the right number of Y indices"
    assert subset_idxs[2].shape[0] == nonzero_pix, "Not the right number of X indices"
    assert np.max(mtx) == 0

def test_cartesian_coords():
    xx, yy = cartesian_coords((0.5, 0.5), (2, 2))
    ref_xx = np.asarray([[-0.5, 0.5], [-0.5, 0.5]])
    ref_yy = np.asarray([[-0.5, -0.5], [0.5, 0.5]])
    assert np.allclose(xx, ref_xx)
    assert np.allclose(yy, ref_yy)

def test_shift2():
    orig = f_test(128)
    outshape = (3 * 128, 3 * 128)
    result = shift2(orig, 0, 0, output_shape=outshape)
    assert result.shape == outshape
    assert np.allclose(orig, result[128:2*128,128:2*128])

    result2 = shift2(orig, -128, -128, output_shape=outshape)
    assert np.allclose(orig, result2[:128,:128])
