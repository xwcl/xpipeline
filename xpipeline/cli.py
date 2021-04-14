import argparse
import logging
import coloredlogs
import os
import sys
import numpy

from dask.distributed import Client

from exao_dap_client.data_store import get_fs
from .commands import compute_sky_model, copy_test, klip, eval_klip, local_klip, diagnostic

from . import utils

log = logging.getLogger(__name__)

COMMANDS = {
    compute_sky_model.ComputeSkyModel,
    copy_test.CopyTest,
    klip.KLIP,
    eval_klip.EvalKLIP,
    local_klip.LocalKLIP,
    diagnostic.Diagnostic,
}

def main():
    parser = argparse.ArgumentParser()
    parser.set_defaults(command_cls=None)
    subps = parser.add_subparsers(title='subcommands',
                                  description='valid subcommands')
    names = set()
    for command_cls in COMMANDS:
        if command_cls.name is None or command_cls.name in names:
            raise Exception(f"Invalid command name for {command_cls}")
        if command_cls.help is None:
            raise Exception(f"Missing help for command {command_cls}")
        subp = subps.add_parser(command_cls.name, help=command_cls.help)
        subp.set_defaults(command_cls=command_cls)
        command_cls.add_arguments(subp)

    args = parser.parse_args()
    if args.command_cls is None:
        parser.print_help()
        sys.exit(1)

    command = args.command_cls(args)
    sys.exit(command.main())
