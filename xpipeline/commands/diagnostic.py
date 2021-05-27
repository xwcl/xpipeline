import logging
import xconf
from .. import version
from .base import BaseCommand

log = logging.getLogger(__name__)

@xconf.config
class Diagnostic(BaseCommand):
    """Power on self-test"""

    def main(self):
        from .. import cli  # avoid circular import
        from .. import tasks, ref

        log.info(f"xpipeline {version.version}")
        command_names = [cls.name for cls in cli.COMMANDS]
        log.info(f"cli commands: {command_names}")
        task_names = [x for x in dir(tasks) if x[0] != "_"]
        log.info(f"task modules: {task_names}")
        ref_names = [x for x in dir(ref) if x[0] != "_"]
        log.info(f"ref modules: {ref_names}")
