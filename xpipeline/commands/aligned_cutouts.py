import sys

import argparse
from xpipeline.core import LazyPipelineCollection
from astropy.utils import data
import fsspec.spec
import logging
import dask
import coloredlogs

from .. import utils

import typing
from .. import xconf
import dataclasses
from dataclasses import dataclass


# from .base import MultiInputCommand

log = logging.getLogger(__name__)

@dataclass
class FileTemplate:
    path : str = xconf.field(help="Path to template FITS image")
    ext : typing.Union[int,str] = xconf.field(default=0, help="Extension containing image data")

@dataclass
class GaussianTemplate:
    fwhm_px : float = xconf.field(default=10, help="Template PSF kernel full-width at half-maximum")

@dataclass
class CutoutConfig:
    search_box_y_ctr : typing.Optional[int]
    search_box_x_ctr : typing.Optional[int]
    search_box_height : typing.Optional[int]
    search_box_width : typing.Optional[int]
    crop_px : int
    template : typing.Union[FileTemplate,GaussianTemplate] = xconf.field(help=utils.unwrap("""
    Template cross-correlated with the search region to align images to a common grid, either given as a FITS image
    or specified as a centered 2D Gaussian with given FWHM
    """))

def _files_from_source(source, extensions):
    # source is a list of directories or files
    # directories should be globbed with a pattern
    # and filenames should be added as-is provided that
    # they exist
    # entries may be either paths or urls

    all_files_paths = []
    for entry in source:
        log.debug(f"Interpreting source entry {entry}")
        fs = utils.get_fs(entry)
        if isinstance(fs, LocalFileSystem):
            # relative paths are only a concern locally
            entry = os.path.realpath(entry)
        if not fs.exists(entry):
            raise RuntimeError(f"Cannot find file or directory {entry}")
        if fs.isdir(entry):
            log.debug(f"Globbing contents of {entry} for {extensions}")
            for extension in extensions:
                if extension[0] == ".":
                    extension = extension[1:]
                glob_result = fs.glob(utils.join(entry, f"*.{extension}"))
                # returned paths from glob won't have protocol string or host
                # so take the basenames of the files and we stick the other
                # part back on from `entry`
                all_files_paths.extend(
                    [utils.join(entry, utils.basename(x)) for x in glob_result]
                )
        else:
            all_files_paths.append(entry)
    # sort file paths lexically
    sorted_files_paths = list(sorted(all_files_paths))
    log.debug(f"Final source files set: {sorted_files_paths} on {fs}")
    if not len(sorted_files_paths):
        raise RuntimeError("Attempting to process empty set of input files")
    return sorted_files_paths

@dataclass
class MultiInputCommand(xconf.Command):
    input : str = xconf.field(help="Input file, directory, or wildcard pattern matching multiple files")
    destination : str = xconf.field(help="Output directory")
    sample_every_n : int = xconf.field(default=1, help="Take every Nth file from inputs (for speed of debugging)")
    file_extensions : list[str] = xconf.field(default=(".fit", ".fits"), help="File extensions to match in the input (when given a directory)")
    ext : typing.Union[str,int] = xconf.field(default=0, help="Extension index or name to load from input files")

    def get_all_inputs(self):
        src_fs = utils.get_fs(self.input)
        if '*' in self.input:
            # handle globbing
            all_inputs = src_fs.glob(self.input)
        else:
            # handle directory
            if src_fs.isdir(self.input):
                all_inputs = []
                for extension in self.file_extensions:
                    glob_result = src_fs.glob(utils.join(self.input, f"*{extension}"))
                    # returned paths from glob won't have protocol string or host
                    # so take the basenames of the files and we stick the other
                    # part back on from `entry`
                    all_inputs.extend(
                        [utils.join(self.input, utils.basename(x)) for x in glob_result]
                    )
            # handle single file
            else:
                all_inputs = [self.input]
        return all_inputs

class AlignedCutouts(MultiInputCommand):
    "Align PSF to template"
    # name = "aligned_cutouts"
    cutouts : dict[str, CutoutConfig] = xconf.field(
        default_factory=lambda: {'default': CutoutConfig(template=GaussianTemplate())},
        help="Specify one or more cutouts with names and template PSFs to generate aligned cutouts for",
    )

    def main(self):
        from .. import pipelines
        from ..tasks import iofits, improc

        log.debug(self)
        dest_fs = utils.get_fs(self.destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {self.destination}")
        dest_fs.makedirs(self.destination, exist_ok=True)

        all_inputs = self.get_all_inputs()
        n_output_files = len(all_inputs)
        output_filepaths = [utils.join(self.destination, f"aligned_cutouts_{i:04}.fits") for i in range(n_output_files)]
        for output_file in output_filepaths:
            if dest_fs.exists(output_file):
                log.error(f"Output exists: {output_file}")
                sys.exit(1)

        input_coll = LazyPipelineCollection(all_inputs)
        coll = input_coll.map(iofits.load_fits_from_path)
        example_hdul = dask.compute(coll.items[0])[0]
        cutout_specs = []
        for name, cutout_config in self.cutouts.items():
            search_box = improc.BBox.from_center(
                center=improc.Pixel(x=cutout_config.search_box_x_ctr, y=cutout_config.search_box_y_ctr),
                extent=improc.PixelExtent(width=cutout_config.search_box_width, height=cutout_config.search_box_height)
            )
            tpl = cutout_config.template
            if isinstance(tpl, GaussianTemplate):
                dimensions = example_hdul[self.ext].data.shape
                center = improc.arr_center(dimensions)
                template_array = improc.gauss2d(dimensions, center, tpl.fwhm_px)
            else:
                hdul = iofits.load_fits_from_path()
                template_array = hdul[tpl.ext].data
            spec = improc.CutoutTemplateSpec(
                search_box=search_box,
                template=template_array,
                name=name
            )
            cutout_specs.append(spec)
        output_coll = pipelines.align_to_templates(coll, cutout_specs)
        return output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True).compute()

