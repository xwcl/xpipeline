import logging
import xconf
import orjson
import numpy as np
from xconf.contrib import BaseRayGrid, FileConfig, join, PathConfig, DirectoryConfig
from ..pipelines.new import MeasureStarlightSubtractionPipeline, StarlightSubtractionMeasurements
from .base import BaseCommand
from pprint import pprint, pformat
import dataclasses

from astropy.io import fits

log = logging.getLogger(__name__)

@xconf.config
class MeasureStarlightSubtraction(BaseCommand, MeasureStarlightSubtractionPipeline):
    destination : DirectoryConfig = xconf.field(default=DirectoryConfig(path="."), help="Directory for output files")  # TODO standardize on directoryconfig
    save_decomposition : bool = xconf.field(default=False, help="Whether to save data decomposition components as a FITS file")
    save_residuals : bool = xconf.field(default=False, help="Whether to save starlight subtraction residuals as a FITS file")
    save_inputs : bool = xconf.field(default=False, help="Whether to save input data (post-injection) and model as a FITS file")
    save_unfiltered_images : bool = xconf.field(default=False, help="Whether to save stacked but unfiltered images")
    save_post_filtering_images : bool = xconf.field(default=False, help="Whether to save stacked and post-filtering images")
    save_ds9_regions : bool = xconf.field(default=False, help="Whether to write a ds9 region file for the signal estimation pixels")

    def main(self):
        if self.save_decomposition:
            self.return_starlight_subtraction = True
            self.subtraction.return_decomposition = True
        if self.save_residuals:
            self.return_starlight_subtraction = True
            self.subtraction.return_residuals = True
        if self.save_inputs:
            self.return_starlight_subtraction = True
            self.subtraction.return_inputs = True
        if self.save_unfiltered_images:
            self.return_starlight_subtraction = True
        if self.save_post_filtering_images:
            self.return_post_filtering_result = True
        if self.save_ds9_regions:
            self.return_post_filtering_result = True
            
        output_filenames = {
            'decomposition.fits': self.save_decomposition,
            'residuals.fits': self.save_residuals,
            'inputs.fits': self.save_inputs,
            'unfiltered.fits': self.save_unfiltered_images,
            'post_filtering.fits': self.save_post_filtering_images,
        }
        self.destination.ensure_exists()
        for fn, condition in output_filenames.items():
            if condition and self.destination.exists(fn):
                log.error(f"Output filename {fn} exists at {self.destination.join(fn)}")
        
        res : StarlightSubtractionMeasurements = self.execute()
        k_modes_values = list(res.by_modes.keys())
        n_inputs = len(self.data.inputs)
        output_dict = self.measurements_to_jsonable(res, k_modes_values)
        log.debug(pformat(output_dict))
        output_json = orjson.dumps(
            output_dict,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
        )
        with self.destination.open_path('result.json', 'wb') as fh:
            fh.write(output_json)
        
        if self.save_unfiltered_images:
            images_by_ext = {}
            for k_modes in k_modes_values:
                for ext, unfiltered_image in res.subtraction_result.modes[k_modes].destination_images.items():
                    if ext not in images_by_ext:
                        images_by_ext[ext] = []
                    images_by_ext[ext].append(unfiltered_image)
            for ext in images_by_ext:
                unfilt_hdul = fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.ImageHDU(np.array(k_modes_values, dtype=int), name="K_MODES_VALUES"),
                ])
                unfilt_hdul.append(fits.ImageHDU(
                    np.stack(images_by_ext[ext]),
                    name=ext
                ))
            with self.destination.open_path('unfiltered.fits', 'wb') as fh:
                unfilt_hdul.writeto(fh)

        if self.save_post_filtering_images or self.save_ds9_regions:
            images_by_filt_by_ext = {}
            kernels_by_filt_by_ext = {}
            for k_modes in k_modes_values:
                for ext, ss_result_by_filter in res.by_modes[k_modes].by_ext.items():
                    for filt_name, ss_result in ss_result_by_filter.items():
                        region_file_name = None
                        if filt_name not in images_by_filt_by_ext:
                            images_by_filt_by_ext[filt_name] = {}
                            kernels_by_filt_by_ext[filt_name] = {}
                        if ext not in images_by_filt_by_ext[filt_name]:
                            images_by_filt_by_ext[filt_name][ext] = []
                            kernels_by_filt_by_ext[filt_name][ext] = []
                        images_by_filt_by_ext[filt_name][ext].append(ss_result.post_filtering_result.image)
                        kernels_by_filt_by_ext[filt_name][ext].append(ss_result.post_filtering_result.kernel)
                        if self.save_ds9_regions and region_file_name is None:
                            from ..tasks import characterization, improc
                            yc, xc = improc.arr_center(images_by_filt_by_ext[filt_name][ext][0])
                            # ds9 is 1-indexed
                            yc += 1
                            xc += 1
                            region_specs = ""
                            kernel_diameter_px = ss_result.post_filtering_result.kernel_diameter_px
                            for idx, (x, y) in enumerate(characterization.simple_aperture_locations(
                                self.data.companion.r_px,
                                self.data.companion.pa_deg,
                                kernel_diameter_px,
                                self.exclude_nearest_apertures,
                                xcenter=xc, ycenter=yc
                            )):
                                region_specs += f"circle({x},{y},{kernel_diameter_px / 2}) # color={'red' if idx == 0 else 'green'}\n"
                            region_file_name = f"regions_{ext}_{filt_name}.reg"
                            with self.destination.open_path(region_file_name, "wb") as fh:
                                fh.write(region_specs.encode('utf8'))
            for ext in images_by_ext:
                postfilt_hdul = fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.ImageHDU(np.array(k_modes_values, dtype=int), name="K_MODES_VALUES"),
                ])
                for filt_name in images_by_filt_by_ext:
                    postfilt_hdul.append(fits.ImageHDU(
                        np.stack(images_by_filt_by_ext[filt_name][ext]),
                        name=f"{ext}_{filt_name}",
                    ))
                    postfilt_hdul.append(fits.ImageHDU(
                        np.stack(kernels_by_filt_by_ext[filt_name][ext]),
                        name=f"{ext}_{filt_name}_KERNEL",
                    ))

            with self.destination.open_path('post_filtering.fits', 'wb') as fh:
                postfilt_hdul.writeto(fh)

        if self.return_starlight_subtraction:
            if self.subtraction.return_decomposition:
                decomp = res.subtraction_result.decomposition
                decomp_hdul = fits.HDUList([fits.PrimaryHDU(),])
                decomp_hdul.append(fits.ImageHDU(decomp.mtx_u0, name='MTX_U0'))
                decomp_hdul.append(fits.ImageHDU(decomp.diag_s0, name='DIAG_S0'))
                decomp_hdul.append(fits.ImageHDU(decomp.mtx_v0, name='MTX_V0'))
                with self.destination.open_path('decomposition.fits', 'wb') as fh:
                    decomp_hdul.writeto(fh)
            if self.subtraction.return_residuals:
                sci_arrays_by_output = [[] for i in range(n_inputs)]
                model_arrays_by_output = [[] for i in range(n_inputs)]
                for k_modes in k_modes_values:
                    for i in range(n_inputs):
                        sci_arrays_by_output[i].append(
                            res.subtraction_result.modes[k_modes].pipeline_outputs[i].sci_arr
                        )
                        model_arrays_by_output[i].append(
                            res.subtraction_result.modes[k_modes].pipeline_outputs[i].model_arr
                        )
                resid_hdul = fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.ImageHDU(np.array(k_modes_values, dtype=int), name="K_MODES_VALUES")
                ])
                for i in range(n_inputs):
                    ext = f"RESID_{i:02}"
                    model_ext = f"MODEL_RESID_{i:02}"
                    sci_array_stack = np.stack(sci_arrays_by_output[i])
                    model_array_stack = np.stack(model_arrays_by_output[i])
                    resid_hdul.append(fits.ImageHDU(sci_array_stack, name=ext))
                    resid_hdul.append(fits.ImageHDU(model_array_stack, name=model_ext))
                with self.destination.open_path('residuals.fits', 'wb') as fh:
                    resid_hdul.writeto(fh)
            if self.subtraction.return_inputs:
                inputs_hdul = fits.HDUList([
                    fits.PrimaryHDU(),
                ])
                for i in range(n_inputs):
                    ext = f"INPUT_{i:02}"
                    model_ext = f"MODEL_INPUT_{i:02}"
                    inputs_hdul.append(fits.ImageHDU(res.subtraction_result.pipeline_inputs[i].sci_arr, name=ext))
                    inputs_hdul.append(fits.ImageHDU(res.subtraction_result.pipeline_inputs[i].model_arr, name=model_ext))
                with self.destination.open_path('inputs.fits', 'wb') as fh:
                    inputs_hdul.writeto(fh)
