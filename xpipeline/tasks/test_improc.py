import numpy as np
import pytest
import dask.array as da
from .. import utils
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

def test_wrap_2d():
    image = np.arange(9).reshape(3, 3)
    mask = np.ones(image.shape, dtype=bool)
    vec = unwrap_image(image, mask)
    wrap_result = wrap_vector(vec, mask)
    assert np.all(image == wrap_result)

def test_unwrap_2d():
    image = np.arange(9, dtype=float).reshape((3, 3))
    mask = np.ones((3, 3), dtype=bool)
    image[1, 1] = np.infty  # make a bogus value to change the max
    mask[1, 1] = False  # mask it out so it doesn't change the max
    mtx = unwrap_image(image, mask)
    assert np.max(mtx) == 8
    assert mtx.shape[0] == 8


def test_cartesian_coords():
    yy, xx = cartesian_coords((0.5, 0.5), (2, 2))
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
    cube_1 = np.arange(9).reshape(3, 3)[np.newaxis, :, :]
    cube_1 = np.repeat(cube_1, 10, axis=0)
    cube_2 = np.arange(9).reshape(3, 3).T[np.newaxis, :, :]
    cube_2 = np.repeat(cube_2, 10, axis=0)
    mask_upper = np.asarray(
        [
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]
    )
    mask_lower = mask_upper.T
    out_cube = improc.combine_paired_cubes(cube_1, cube_2, mask_upper, mask_lower)
    ref = np.array([[[np.nan, 1.0, 2.0], [1.0, np.nan, 5.0], [2.0, 5.0, np.nan]]])
    # can't compare nan == nan, so...
    assert np.all(np.nan_to_num(ref) == np.nan_to_num(out_cube))

    # change fill value
    out_cube = improc.combine_paired_cubes(
        cube_1, cube_2, mask_upper, mask_lower, fill_value=0
    )
    ref = np.array([[[0, 1.0, 2.0], [1.0, 0, 5.0], [2.0, 5.0, 0]]])
    assert np.all(ref == out_cube)

def test_derotate_cube():
    data = np.zeros((3, 3, 3))
    data[0, 1, 0] = 1
    data[1, 2, 1] = 1
    data[2, 1, 2] = 1
    data = np.asarray(data)
    angles = np.asarray([-90, 0, 90])
    out_cube = improc.derotate_cube(data, angles, fill_value=0)
    assert np.all(out_cube[:, 2, 1] > 0.99)


def test_aligned_cutout_oversized_template():
    picshape = 128, 128
    psfim = improc.gauss2d(picshape, improc.arr_center(picshape), (10, 10))
    sci_arr = improc.ft_shift2(psfim, -4.33, -5.75)[15:100,20:90]
    spec = improc.CutoutTemplateSpec(
        search_box=improc.BBox(origin=improc.Pixel(0,0), extent=improc.PixelExtent(*picshape)),
        template=psfim,
        name="primary"
    )
    res = improc.aligned_cutout(sci_arr, spec)
    assert np.nanmean((res - psfim)[15:100,20:90]) < 1e-5

def test_aligned_cutout_undersized_template():
    picshape = 128, 128
    psfim = improc.gauss2d(picshape, improc.arr_center(picshape), (10, 10))
    sci_arr = improc.ft_shift2(psfim, -4.33, -5.75)
    spec = improc.CutoutTemplateSpec(
        search_box=improc.BBox(origin=improc.Pixel(0,0), extent=improc.PixelExtent(*picshape)),
        template=psfim[5:-5,5:-5],
        name="primary"
    )
    res = improc.aligned_cutout(sci_arr, spec)
    assert np.nanmean((res - psfim[5:-5,5:-5])) < 1e-5

def test_rotate():
    # cpu
    image = improc.f_test(100)
    # no dest supplied
    result1 = improc.rotate(image, 45)
    # just smoke test:
    assert np.count_nonzero(result1) != 0, "CPU rotate somehow clobbered input"
    # yes dest supplied
    dest = np.zeros_like(image)
    result2 = improc.rotate(image, 45, dest_image=dest)
    assert result2 is dest, "CPU rotate supplying dest array didn't return the dest array"
    nanmask = np.isnan(result1)
    assert np.allclose(result1[~nanmask], result2[~nanmask]), "CPU rotate with supplied dest array produced different answer from newly allocated"
    # 90deg agrees
    # nb sense of angle reversed between this code and np/cp.rot90
    result3 = improc.rotate(image, -90, fill_value=0.0)  # TODO flip arrays first and only rotate the remaining <90deg
    assert np.allclose(result3, np.rot90(image)), "CPU interpolated image disagrees with simple 90deg rotation"

def test_make_rotation_about_center():
    shape = 10, 10
    ctr_x, ctr_y = 4.5, 4.5
    rotation_deg = 45
    xform = improc.translation_matrix(ctr_x, ctr_y) @ improc.rotation_matrix(np.deg2rad(-rotation_deg)) @ improc.translation_matrix(-ctr_x, -ctr_y)
    assert np.allclose(xform, improc.make_rotation_about_center(shape, rotation_deg))

def test_matrix_transform_cube():
    npix = 50
    cube = np.array([
        improc.f_test(npix),
        np.rot90(improc.f_test(npix)),
        np.rot90(np.rot90(improc.f_test(npix)))
    ])
    transforms = np.array([
        improc.make_rotation_about_center((npix, npix), 0),
        improc.make_rotation_about_center((npix, npix), -90),
        improc.make_rotation_about_center((npix, npix), -180),
    ])
    outcube = np.zeros_like(cube)
    improc.matrix_transform_cube(cube, transforms, outcube, 0)
    assert np.allclose(cube[0], outcube[0])


def test_max_radius():
    npix = 128
    ctr = (npix - 1) / 2
    assert improc.max_radius((ctr, ctr), (npix, npix)) == 63.5

def test_encircled_energy_radial_profile():
    test_data = np.ones((64, 64))
    ee_r, ee, profile_r, profile = improc.encircled_energy_and_profile(test_data, improc.arr_center(test_data))
    assert np.abs(np.pi * ee_r[-1]**2 - ee[-1]) < 1, "Encircled energy disagrees with area formula"
    profile_values = np.unique(profile)
    assert len(profile_values) == 1, "Profile should be identically 1 at all radii"
    assert profile_values[0] == 1, "Profile should be identically 1 at all radii"

@pytest.mark.skipif(
    not utils.version_greater_or_equal(np.__version__, "1.21"),
    reason="period= arg to unwrap arrived in NumPy 1.21"
)
def test_combine_ranges():
    data = np.zeros((6, 3, 3))
    angles = np.zeros(6)

    data[0,0,0] = 1
    data[1,0,0] = 1
    data[2,0,0] = 1

    data[3,1,0] = 1
    data[4,1,0] = 1
    data[5,1,0] = 1

    idxspec = improc.FrameIndexRangeSpec(n_frames=3)
    anglespec = improc.AngleRangeSpec(delta_deg=90, angle_deg_column_name="meta3")
    rpxspec = improc.PixelRotationRangeSpec(delta_px=np.pi / 2, r_px=1, angle_deg_column_name="meta3")
    metadata = np.array([
        ("A", 1, 90.0),
        ("B", 2, 60.0),
        ("C", 3, 30.0),
        ("D", 4, 0.0),
        ("E", 5, -30.0),
        ("F", 6, -60.0),
    ], dtype=[("meta1", "S1"), ("meta2", np.int32), ("meta3", np.float32),])
    for spec in (idxspec, anglespec, rpxspec):
        print(f'{spec=}')
        final_seqs, final_metadata = improc.combine_ranges([data, data], metadata, spec)
        for final_cube in final_seqs:
            final_angles = final_metadata['meta3']
            print(f"{final_cube=}")
            print(f"{final_metadata=}")
            assert final_cube.shape[0] == 2
            assert final_cube[0,0,0] == 1
            assert np.isclose(final_angles[0], 60)
            assert final_metadata['meta1'][0] == b'A'
            assert np.isclose(final_metadata['meta3'][0], 60)
            assert final_cube[1,1,0] == 1
            assert np.isclose(final_angles[1], -30)
            assert final_metadata['meta1'][1] == b'D'
            assert np.isclose(final_metadata['meta3'][1], -30)

def test_histogram_std():
    xs = improc.pixel_centers(100)
    sigma = 10
    ys = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-xs**2 / (2 * sigma**2))
    est_std = improc.histogram_std(ys)
    assert (est_std - sigma) < 0.005

def test_radial_stds_cube():
    N = 100
    cube = np.zeros((3, N, N))
    frame_shape = cube[0].shape
    cube[0] = improc.gauss2d(frame_shape, improc.arr_center(frame_shape), (5, 2))
    cube[1] = improc.gauss2d(frame_shape, improc.arr_center(frame_shape), (2, 5))
    cube[2] = improc.gauss2d(frame_shape, improc.arr_center(frame_shape), (1, 1))
    radial_stds = improc.radial_stds_cube(cube)
    assert np.isclose(radial_stds[0], radial_stds[1]), "y,x and x,y sigmas giving different answers"
    assert np.isclose(radial_stds[1], np.sqrt(29))
    assert np.isclose(radial_stds[2], np.sqrt(2))
