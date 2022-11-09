import os
from argparse import ArgumentParser
from tqdm import tqdm
import yaml

from torch.optim import Adam
import numpy as np

from models.fno import FNO1d

def train(config, args):
    pass



if __name__ == '__main__':
    parser = ArgumentParser('Parser for training FNO')
    parser.add_argument('--datapath', type=str, default='data/poisson/train-s100.pickle')
    parser.add_argument('--logdir', type=str, default='exp/poisson/default')
    parser.add_argument('--config', type=str, default='configs/poisson/fno.yaml')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    train()
