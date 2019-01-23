import os
import sys
import time
import datetime
import torch
import torch.nn as nn
import argparse

from util import get_config, Logger, Checkpoint
from trainer import Trainer

def main():
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--tag', default=None, type=str, help='Experiment tag')
    parser.add_argument('--preset', choices=['gan', 'syn', 'mod', 'gan+syn'], default='gan', type=str, required=True, help='Training configuration preset')
    parser.add_argument('--clear_cache', action='store_true', help='Remove all cache files')
    parser.add_argument('--set', default=None, type=str, nargs=argparse.REMAINDER, help='Optional settings')
    args = parser.parse_args()

    cfg = get_config(args, now, os.getcwd())
    checkpoint = Checkpoint(cfg)

    logger = Logger(cfg, checkpoint.tensorboard_path)
    logger.write('Now: ' + now)

    Trainer(cfg, checkpoint, logger).train()

    logger.close()

if __name__ == '__main__':
    main()    
