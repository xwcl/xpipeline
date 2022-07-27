import logging
import xconf
from xconf.contrib import DirectoryConfig
from ..pipelines.new import (
    SaveMeasuredStarlightSubtraction,
    MeasureStarlightSubtraction,
    StarlightSubtractionDataConfig
)
from .base import BaseCommand

log = logging.getLogger(__name__)

@xconf.config
class SaveMeasuredStarlightSubtraction(BaseCommand, SaveMeasuredStarlightSubtraction):
    destination : DirectoryConfig = xconf.field(default=DirectoryConfig(path="."), help="Directory for output files")  # TODO standardize on directoryconfig
    measure_subtraction : MeasureStarlightSubtraction = xconf.field(help="Configure starlight subtraction and measurement")
    data : StarlightSubtractionDataConfig = xconf.field(help="Starlight subtraction data and injected companion configuration")

    def main(self):
        data = self.data.load()
        self.execute(data, self.measure_subtraction, self.destination)