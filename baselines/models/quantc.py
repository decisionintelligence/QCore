import torch, pandas as pd, numpy as np

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from quant.train import calibrate_flip


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='QuantCore')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class QuantC(ContinualModel):
    NAME = 'quantc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(QuantC, self).__init__(backbone, loss, args, transform)
        self.args = args
        #self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, trainset, streamset, not_aug_inputs):
        return calibrate_flip(self.args, self.net, trainset, streamset)
