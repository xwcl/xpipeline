import numpy as np
import pytest
import dask.array as da
from .. import core
from . import improc
from .improc import (
    rough_peak_in_box,
    unwrap_cube,
    unwrap_image,
    wrap_matrix,
    wrap_vector,
    cartesian_coords,
    f_test,
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


@pytest.mark.parametrize("xp", [np, da])
def test_wrap_2d(xp):
    image = xp.arange(9).reshape(3, 3)
    mask = np.ones(image.shape, dtype=bool)
    vec, subset_idxs = unwrap_image(image, mask)
    wrap_result = wrap_vector(vec, image.shape, subset_idxs)
    if xp is da:
        image = image.compute()
        wrap_result = wrap_result.compute()
    assert np.all(image == wrap_result)


@pytest.mark.parametrize("xp", [np, da])
def test_wrap_3d(xp):
    imcube = xp.arange(27).reshape(3, 3, 3)
    mask = np.ones(imcube.shape, dtype=bool)
    vec, subset_idxs = unwrap_image(imcube, mask)
    wrap_result = wrap_vector(vec, imcube.shape, subset_idxs)
    if xp is da:
        imcube = imcube.compute()
        wrap_result = wrap_result.compute()
    assert np.all(imcube == wrap_result)

    imcube = xp.arange(27).reshape(3, 3, 3)
    imcube = xp.repeat(imcube[core.newaxis, :, :, :], 3, axis=0)
    mtx, subset_idxs = unwrap_cube(imcube, mask)
    wrap_result = wrap_matrix(mtx, imcube.shape, subset_idxs)
    if xp is da:
        imcube = imcube.compute()
        wrap_result = wrap_result.compute()
    assert np.all(imcube == wrap_result)


@pytest.mark.parametrize("xp", [np, da])
def test_unwrap_2d(xp):
    image = np.arange(9, dtype=float).reshape((3, 3))
    mask = np.ones((3, 3), dtype=bool)
    image[1, 1] = np.infty  # make a bogus value to change the max
    if xp is da:
        image = da.from_array(image)
    mask[1, 1] = False  # mask it out so it doesn't change the max
    mtx, subset_idxs = unwrap_image(image, mask)
    if xp is da:
        mtx = mtx.compute()
    assert np.max(mtx) == 8
    assert mtx.shape[0] == 8
    assert subset_idxs[1].shape[0] == 8, "Not the right number of X indices"
    assert subset_idxs[0].shape[0] == 8, "Not the right number of Y indices"


@pytest.mark.parametrize("xp", [np, da])
def test_unwrap_one_3d(xp):
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


@pytest.mark.parametrize("xp", [np, da])
def test_unwrap_many_3d(xp):
    imcube = np.zeros((3, 3, 3))
    mask = np.ones(imcube.shape, dtype=bool)

    # set one nonzero pixel, and mask it out so we can tell if
    # masking worked when it's not present in the output
    imcube[1, 1, 1] = 1
    mask[1, 1, 1] = False
    if xp is da:
        imcube = da.from_array(imcube)
    imcube = xp.repeat(imcube[core.newaxis, :, :, :], 3, axis=0)
    mtx, subset_idxs = unwrap_cube(imcube, mask)
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


def test_ft_shift2():
    orig = f_test(128)
    assert np.allclose(orig, improc.ft_shift2(orig, 0, 0))
    outshape = (3 * 128, 3 * 128)
    result = improc.ft_shift2(orig, 0, 0, output_shape=outshape)
    assert result.shape == outshape
    assert np.allclose(orig, result[128 : 2 * 128, 128 : 2 * 128])

    result2 = improc.ft_shift2(orig, -128, -128, output_shape=outshape)
    assert np.allclose(orig, result2[:128, :128])

    restored = orig
    for i in range(20):  # error shouldn't accumulate quickly
        shifted = improc.ft_shift2(restored, 0.5, 0.5)
        restored = improc.ft_shift2(shifted, -0.5, -0.5)
    assert np.allclose(orig, restored)


def test_shift2():
    orig = f_test(128)
    outshape = (3 * 128, 3 * 128)
    result = improc.shift2(orig, 0, 0, output_shape=outshape)
    assert result.shape == outshape
    assert np.allclose(orig, result[128 : 2 * 128, 128 : 2 * 128])

    result2 = improc.shift2(orig, -128, -128, output_shape=outshape)
    assert np.allclose(orig, result2[:128, :128])


@pytest.mark.parametrize("xp", [np, da])
def test_combine_paired_cubes(xp):
    cube_1 = xp.arange(9).reshape(3, 3)[core.newaxis, :, :]
    cube_1 = xp.repeat(cube_1, 10, axis=0)
    cube_2 = xp.arange(9).reshape(3, 3).T[core.newaxis, :, :]
    cube_2 = xp.repeat(cube_2, 10, axis=0)
    mask_upper = np.asarray(
        [
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]
    )
    mask_lower = mask_upper.T
    out_cube = improc.combine_paired_cubes(cube_1, cube_2, mask_upper, mask_lower)
    if xp is da:
        out_cube = out_cube.compute()
    ref = np.array([[[np.nan, 1.0, 2.0], [1.0, np.nan, 5.0], [2.0, 5.0, np.nan]]])
    # can't compare nan == nan, so...
    assert np.all(np.nan_to_num(ref) == np.nan_to_num(out_cube))

    # change fill value
    out_cube = improc.combine_paired_cubes(
        cube_1, cube_2, mask_upper, mask_lower, fill_value=0
    )
    if xp is da:
        out_cube = out_cube.compute()
    ref = np.array([[[0, 1.0, 2.0], [1.0, 0, 5.0], [2.0, 5.0, 0]]])
    assert np.all(ref == out_cube)


@pytest.mark.parametrize("xp", [np, da])
def test_derotate_cube(xp):
    data = np.zeros((3, 3, 3))
    data[0, 1, 0] = 1
    data[1, 2, 1] = 1
    data[2, 1, 2] = 1
    data = xp.asarray(data)
    if xp is da:
        data = da.repeat(data, 10, axis=0)
    angles = xp.asarray([-90, 0, 90])
    if xp is da:
        angles = da.repeat(angles, 10, axis=0)
    out_cube = improc.derotate_cube(data, angles)
    if xp is da:
        out_cube = out_cube.compute()
    assert np.all(out_cube[:, 2, 1] > 0.99)

def test_aligned_cutout():
    picshape = 128, 128
    psfim = improc.gauss2d(picshape, improc.center(picshape), (10, 10))
    sci_arr = improc.ft_shift2(psfim, 15.75, 13.5)
    spec = improc.CutoutTemplateSpec(origin=(0, 0), extent=picshape, template=psfim, name='primary')
    res = improc.aligned_cutout(sci_arr, spec)
    assert np.average((res - psfim)[25:100,25:100]) < 1e-5
