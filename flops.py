from __future__ import print_function

import sys
import os
import argparse
import numpy as np
if '/data/software/opencv-3.4.0/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.4.0/lib/python2.7/dist-packages')
if '/data/software/opencv-3.3.1/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.3.1/lib/python2.7/dist-packages')
import cv2
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from lib.utils.config_parse import cfg_from_file
from lib.ssds_train import Solver
from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
            help='optional config file', default=None, type=str)
    parser.add_argument('--multi_gpu', dest='multi_gpu', help='Use multi-gpu training or testing', action='store_true')


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def test():
    with torch.cuda.device(0):
        with torch.no_grad():
            args = parse_args()
            if args.config_file is not None:
                cfg_from_file(args.config_file)
        #test_model()
            s = Solver(args)
            model = s.model
            batch = torch.FloatTensor(1, 3, 300, 300).cuda(0)

            model = add_flops_counting_methods(model)
            model.eval().start_flops_count()
            out = model(batch)

            print(model)
        #print('Output shape: {}'.format(list(out.shape)))
            print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
            print('Params: ' + get_model_parameters_number(model))


def test_flop_convdw():
    with torch.cuda.device(0):
        model = _conv_dw(3, 64, stride=2, padding=1, expand_ratio=1)

        batch = torch.FloatTensor(1, 3, 300, 300).cuda(0)

        model = add_flops_counting_methods(model)
        model.eval().start_flops_count()
        out = model(batch)

        print(model)
    #print('Output shape: {}'.format(list(out.shape)))
        print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
        print('Params: ' + get_model_parameters_number(model))


def _conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

if __name__ == '__main__':
    test()
