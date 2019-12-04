import re
from . import color as C
import logging
import os
import sys

def is_number(token):
    num_regex = re.compile(r'^[+-]?[0-9]+\.?[0-9]*$')
    return bool(num_regex.match(token))


def set_logger(out_dir=None, fn='log.txt'):
    console_format = C.BColors.OKBLUE + '[%(levelname)s]' + C.BColors.ENDC + \
        '%(asctime)s  (%(name)s) %(message)s'
    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console)
    if out_dir:
        file_format = '[%(levelname)s] %(asctime)s (%(name)s) %(message)s'
        log_file = logging.FileHandler(os.path.join(out_dir, fn), mode='w')
        log_file.setLevel(logging.INFO)
        log_file.setFormatter(logging.Formatter(file_format))
        logger.addHandler(log_file)


def print_configs(config, path=None):
    logger = logging.getLogger(__name__)
    logger.info("Configurations:")
    if path:
        output_file = open(path, 'w')
    for key, value in sorted(config.items()):
        info = f"\t{key}: {value}"
        logger.info(info)
        if path:
            output_file.write(info+'\n')
    if path:
        output_file.close()


def print_args(args, path=None):
    if path:
        output_file = open(path, 'w')
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    args.command = ' '.join(sys.argv)
    items = vars(args)
    for key in sorted(items.keys(), key=lambda s: s.lower()):
        value = items[key]
        if not value:
            value = "None"
        logger.info("  " + key + ": " + str(items[key]))
        if path is not None:
            output_file.write("  " + key + ": " + str(items[key]) + "\n")
    if path:
        output_file.close()
    del args.command
