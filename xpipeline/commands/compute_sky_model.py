import logging
import xconf

from .base import MultiInputCommand

log = logging.getLogger(__name__)


@xconf.config
class ComputeSkyModel(MultiInputCommand):
    """Compute sky model eigenimages"""
    n_components : int = xconf.field(default=6, help="Number of PCA components to calculate")
    mask_dilate_iters : int = xconf.field(default=6, help="Number of times to grow mask regions before selecting cross-validation pixels")
    test_fraction : float = xconf.field(default=0.25, help="Fraction of inputs to reserve for cross-validation")

    def main(self):
        import dask
        from .. import utils
        from .. import pipelines
        from ..core import LazyPipelineCollection
        from ..tasks import iofits

        # outputs
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        log.debug(f"calling makedirs on {dest_fs} at {self.destination}")
        dest_fs.makedirs(destination, exist_ok=True)
        model_fn = utils.join(self.destination, "sky_model.fits")
        self.quit_if_outputs_exist([model_fn])

        # execute
        inputs_coll = LazyPipelineCollection(self.get_all_inputs()).map(iofits.load_fits_from_path)
        one_input_hdul = dask.compute(inputs_coll.items[0])[0]
        plane_shape = one_input_hdul[0].data.shape

        d_sky_model = pipelines.compute_sky_model(
            inputs_coll,
            plane_shape,
            self.test_fraction,
            self.random_state,
            self.n_components,
            self.mask_dilate_iters,
        )
        the_sky_model = dask.compute(d_sky_model)[0]
        hdul = the_sky_model.to_hdulist()
        hdul.writeto(model_fn, overwrite=True)
        log.info(f"Sky model written to {model_fn}")
