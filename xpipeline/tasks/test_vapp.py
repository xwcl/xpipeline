from . import vapp, improc
import astropy.units as u
from ..ref import clio


def test_make_dark_hole_masks():
    shape = (167, 167)
    ctr = shape[0] // 2
    top_mask, bot_mask = vapp.make_dark_hole_masks(
        shape,
        owa_px=clio.lambda_over_d_to_pixel(clio.VAPP_OWA_LAMD, 3.9 * u.um).value,
        offset_px=clio.lambda_over_d_to_pixel(clio.VAPP_OFFSET_LAMD, 3.9 * u.um).value,
        psf_rotation_deg=clio.VAPP_PSF_ROTATION_DEG,
    )
    assert bot_mask[ctr, ctr] == 0
    assert top_mask[ctr, ctr] == 0
    assert bot_mask[ctr - 25, ctr - 25] == False
    assert top_mask[ctr - 25, ctr - 25] == True

    assert bot_mask[ctr + 25, ctr + 25] == True
    assert top_mask[ctr + 25, ctr + 25] == False


def test_mask_along_angle():
    left_mask, right_mask = vapp.mask_along_angle((2, 2), 0)
    assert left_mask[0, 0] == 1
    assert right_mask[0, 0] == 0
    assert left_mask[0, 1] == 0
    assert right_mask[0, 1] == 1

    left_mask, right_mask = vapp.mask_along_angle((2, 2), 90)
    assert left_mask[0, 0] == 1
    assert right_mask[0, 0] == 0
    assert left_mask[1, 0] == 0
    assert right_mask[1, 0] == 1
