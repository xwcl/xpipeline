import logging
from typing import Optional
import xconf
from .. import utils
from .base import InputCommand
from .base import CompanionConfig, TemplateConfig

log = logging.getLogger(__name__)

@xconf.config
class Inject(InputCommand):
    "Inject companion signals into a sequence of observations"
    templates : dict[str,TemplateConfig] = xconf.field(help="Paths for template data and scale factors")
    companions : Optional[list[CompanionConfig]] = xconf.field(help="Locations and scales of injected/anticipated companion signals")
    saturation_threshold : float = xconf.field(default=float('inf'), help="Upper limit at which pixel values will be clipped")

    def _load_dataset(self, dataset_path):
        from ..tasks import iofits
        input_cube_hdul = iofits.load_fits_from_path(dataset_path)
        obs_method = utils.parse_obs_method(input_cube_hdul[0].header["OBSMETHD"])
        return input_cube_hdul, obs_method

    def main(self):
        from ..tasks import iofits
        output_injected_data_fn = utils.join(self.destination, "injected_dataset.fits")
        outputs = [output_injected_data_fn]
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)
        self.quit_if_outputs_exist(outputs)

        dataset_hdul, obs_method = self._load_dataset(self.input)
        companion_specs = self._get_companion_specs()
        templates = self._load_templates()
        dataset_hdul = self._inject_companions(dataset_hdul, obs_method, templates, companion_specs)
        iofits.write_fits(dataset_hdul, output_injected_data_fn)

    def _get_companion_specs(self):
        from ..tasks import characterization
        specs = []
        for companion in self.companions:
            specs.append(characterization.CompanionSpec(
                scale=companion.scale,
                r_px=companion.r_px,
                pa_deg=companion.pa_deg
            ))
        return specs
    
    def _load_templates(self):
        from ..tasks import iofits, characterization
        templates = {}
        for name, template_config in self.templates.items():
            templates_hdul = iofits.load_fits_from_path(template_config.path)
            template_ext = template_config.ext if template_config.ext is not None else name
            if template_config.scale_factors_path is not None:
                scales_hdul = iofits.load_fits_from_path(template_config.scale_factors_path)
                scale_ext = template_config.scale_factors_ext if template_config.scale_factors_ext is not None else name
                scale_factors = scales_hdul[scale_ext].data
            else:
                scale_factors = 1.0
                scale_ext = None
            
            templates[name] = characterization.TemplateSignal(
                templates_hdul[template_ext].data,
                scale_factors
            )
            log.info(f"Loaded {template_config.path=} {template_ext=} {template_config.scale_factors_path=} {scale_ext=} into new TemplateSignal")
        return templates

    def _get_derotation_angles(self, dataset_hdul, obs_method):
        if "adi" in obs_method:
            where_angles = obs_method["adi"]["derotation_angles"]
            if '.' in where_angles:
                tbl_ext, col_name = where_angles.split('.', 1)
                derotation_angles = dataset_hdul[tbl_ext].data[col_name]
            else:
                derotation_angles = dataset_hdul[where_angles].data
        else:
            derotation_angles = None
        return derotation_angles


    def _inject_companions(self, dataset_hdul, obs_method, templates, companion_specs):
        from ..tasks import iofits, characterization

        log.info("Injecting signals")
        derotation_angles = self._get_derotation_angles(dataset_hdul, obs_method)
        if "vapp" in obs_method:
            left_extname = obs_method["vapp"]["left"]
            right_extname = obs_method["vapp"]["right"]
            injectable_extnames = [left_extname, right_extname]
        elif "ext" in obs_method:
            injectable_extnames = [obs_method["ext"]]
        else:
            raise RuntimeError("Nowhere to inject")

        for extname in injectable_extnames:
            dataset_hdul[extname].data, signal_only = characterization.inject_signals(
                dataset_hdul[extname].data,
                companion_specs,
                templates[extname].signal,
                angles=derotation_angles,
                template_scale_factors=templates[extname].scale_factors,
                saturation_threshold=self.saturation_threshold
            )
            log.debug(f"Injected {extname}")
            dataset_hdul.append(iofits.DaskHDU(signal_only, name=f"{extname}_SIGNAL"))
        return dataset_hdul
