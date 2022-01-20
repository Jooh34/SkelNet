import argparse
import os
import json
import copy
import shutil
from types import SimpleNamespace as Namespace
import torch
from model.skel import skel_net

from trainer.trainer import skel_trainer 

def train(config):
    trainer = skel_trainer(config)
    trainer.train()

    net = skel_net(config)

def parse_config(args):
    config = copy.deepcopy(json.load(open(args.config), object_hook=lambda d: Namespace(**d)))
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
        default='./configs/default.json',
        type=str, help='config file (default: default.json)'
    )

    seed = 1
    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    args = parser.parse_args()
    if args.config:
        config = parse_config(args)
    else:
        AssertionError("missing config file path.")

    if os.path.exists(config.trainer.ckpt_dir) and 'debug' not in args.name and args.resume is None:
        allow_cover = input('Model file detected, do you want to replace it? (Y/N)').lower()
        if allow_cover == 'n':
            exit()
        else:
            shutil.rmtree(config.trainer.ckpt_dir, ignore_errors=True)

    train(config)

if __name__ == '__main__':
    main()