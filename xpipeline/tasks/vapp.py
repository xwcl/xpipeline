import numpy as np
from . import improc


def make_dark_hole_masks(shape, owa_px, offset_px, psf_rotation_deg, maximize_iwa=False):
    """Generate dark hole masks for a gvAPP-180 using OWA, center
    offset, and rotation

    Parameters
    ----------
    shape : tuple
    owa_px : float
        Outer working angle measured in pixels from the *shifted* center
    offset_px : float
        Center location shift in -X for left and +X for right (adjusted
        by `psf_rotation_deg`)
    psf_rotation_deg : float
        Amount by which the flat edges of the masks (and the offset
        from center) are rotated E of N (CCW when 0,0 at lower left)
    maximize_iwa : bool
        Whether included region should contain bright PSF structure
        all the way to the `psf_rotation_deg` line dividing the image

    Returns
    -------
    left_mask : np.ndarray
        Mask that is True within the dark hole that lies on the left
        side when `psf_rotation_deg` == 0
    right_mask : np.ndarray
        Mask that is True within the dark hole that lies on the right
        side when `psf_rotation_deg` == 0
    """
    radius_px = owa_px + offset_px

    right_overall_rotation_radians = np.deg2rad(-psf_rotation_deg + 90)
    right_offset_theta = np.deg2rad(psf_rotation_deg)
    ctr_x, ctr_y = (shape[1] - 1) / 2, (shape[0] - 1) / 2
    right_mask = improc.mask_arc(
        (
            ctr_y + offset_px * np.sin(right_offset_theta),
            ctr_x + offset_px * np.cos(right_offset_theta),
        ),
        shape,
        from_radius=0,
        to_radius=radius_px,
        from_radians=0,
        to_radians=np.pi,
        overall_rotation_radians=right_overall_rotation_radians,
    )
    left_mask = improc.mask_arc(
        (
            ctr_y - offset_px * np.sin(right_offset_theta),  # swap offset signs since it's left
            ctr_x - offset_px * np.cos(right_offset_theta),
        ),
        shape,
        from_radius=0,
        to_radius=radius_px,
        from_radians=0,
        to_radians=np.pi,
        overall_rotation_radians=right_overall_rotation_radians + np.pi,
    )
    if maximize_iwa:
        bright_bar = improc.mask_box(shape, (ctr_x, ctr_y), (offset_px * 2, 2 * radius_px), -psf_rotation_deg)
        left_half, right_half = mask_along_angle(shape, psf_rotation_deg)
        left_mask = (left_mask | bright_bar) & left_half
        right_mask = (right_mask | bright_bar) & right_half

    return left_mask, right_mask


def mask_along_angle(shape, deg_e_of_n):
    """Generate complementary masks for a gvAPP-180 by dividing the
    image along a line through the center at an angle `deg_e_of_n`

    Parameters
    ----------
    shape : tuple
    deg_e_of_n : float
        Amount by which the dividing line is rotated E of N
        (CCW when 0,0 at lower left)

    Returns
    -------
    left_half : np.ndarray
        Mask that is True for pixels left of the dividing line
        when `deg_e_of_n` == 0
    right_half : np.ndarray
        Mask that is True for pixels right of the dividing line
        when `deg_e_of_n` == 0
    """
    _, theta = improc.polar_coords(improc.arr_center(shape), shape)
    left_half = (theta < np.deg2rad(-90 + deg_e_of_n)) | (
        (theta > np.deg2rad(90 + deg_e_of_n)) & (theta > 0)
    )
    right_half = ~left_half
    return left_half, right_half


def _count_nans(arr):
    return np.count_nonzero(np.isnan(arr))


def determine_frames_crop_amount(frame_a, frame_b):
    the_slice = slice(0, frame_a.shape[1])
    crop_px = 0
    while (
        _count_nans(frame_a[the_slice, the_slice]) != 0
        or _count_nans(frame_b[the_slice, the_slice]) != 0
    ):
        crop_px += 1
        the_slice = slice(crop_px, -crop_px)
    return crop_px


def determine_cubes_crop_amount(left_cube, right_cube):
    # note *not* nansum, since we want nans to propagate
    left_combined = np.sum(left_cube, axis=0)
    right_combined = np.sum(right_cube, axis=0)
    return determine_frames_crop_amount(left_combined, right_combined)


def crop_paired_frames(left_frame, right_frame):
    crop_px = determine_frames_crop_amount(left_frame, right_frame)
    if crop_px > 0:
        return (
            left_frame[crop_px:-crop_px, crop_px:-crop_px],
            right_frame[crop_px:-crop_px, crop_px:-crop_px],
        )
    return left_frame, right_frame


def crop_paired_cubes(left_cube, right_cube):
    crop_px = determine_cubes_crop_amount(left_cube, right_cube)
    if crop_px > 0:
        cropped_left_cube = left_cube[:, crop_px:-crop_px, crop_px:-crop_px]
        cropped_right_cube = right_cube[:, crop_px:-crop_px, crop_px:-crop_px]
        return cropped_left_cube, cropped_right_cube
    return left_cube, right_cube
