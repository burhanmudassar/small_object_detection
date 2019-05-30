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
from lib.utils.timer import Timer
from lib.utils.config_parse import cfg

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


def _forward_features_size(model, img_size):
    model.eval()
    x = torch.rand(1, 3, img_size[0], img_size[1])
    x = torch.autograd.Variable(x, volatile=True)  # .cuda()
    feature_maps = model(x, phase='feature')
    return [(o.size()[2], o.size()[3]) for o in feature_maps]

def test():
    with torch.cuda.device(0):
        with torch.no_grad():
            args = parse_args()
            if args.config_file is not None:
                cfg_from_file(args.config_file)
        #test_model()
            s = Solver(args)
            model = s.model
            _t = Timer()

            batch_size = 16

            timing_array = []
            for i in range(1000):

                _t.tic()
                batch = torch.FloatTensor(batch_size, 3, cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]).cuda(0)
                model = add_flops_counting_methods(model)
                model.eval().start_flops_count()
                out = model(batch)
                inf_time = _t.toc()
                timing_array.append(inf_time)

            print("Inference Time Mean: {:0.6f} Std Dev: {:0.6f}".format(np.mean(timing_array)*1000/batch_size, np.std(timing_array)*1000/batch_size))

            #print(model)


        #print('Output shape: {}'.format(list(out.shape)))
            print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
            print('Params: ' + get_model_parameters_number(model))




if __name__ == '__main__':
    test()
