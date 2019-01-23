import os
from glob import glob
import numpy as np
import cv2
import torch
import torchvision
from torch import nn
import datetime
from tqdm import trange
import argparse
import re
import math
import time
import shutil

from util import get_config, Logger, Checkpoint, compute_auroc
from data import imdb, dataset
from model.senet import senet154

class DatasetTest(dataset.DatasetGAN):

    def _jpeg(self, face):
        return face
    
    def _crop_random(self, idx):
        return self._crop_center(idx)

@torch.no_grad()
def test(cfg, logger, root_result):
    os.makedirs(root_result, exist_ok=True)
    root_dset = os.path.abspath(cfg.root_dset)
    root_model = os.path.abspath(cfg.root_model)

    print('getting imglist...')
    imglist = [os.path.join('test', _) for _ in os.listdir(os.path.join(cfg.data_path, 'test')) if os.path.splitext(_)[1].lower() in ['.jpg', '.png', '.jpeg']]
    imglist.sort(key=lambda x: os.path.basename(x))
    print('got {} imgs'.format(len(imglist)))
    assert len(imglist) > 0

    _imdb = imdb.Imdb(cfg, name='Testset', virtual=True, logger=logger)
    _imdb.imgpath = np.array(imglist)
    _imdb.bbox_original = np.array([[0, 0, 128, 128]] * len(imglist), dtype=np.int32)
    _imdb.bbox_min = np.copy(_imdb.bbox_original)
    _imdb.bbox_max = np.copy(_imdb.bbox_original)
    _imdb.minsize = np.array([128] * len(imglist), dtype=np.int32)
    _imdb.maxsize = np.array([128] * len(imglist), dtype=np.int32)
    _dset = DatasetTest(cfg, _imdb, 'train', 'real')

    result = list()
    model_list = glob(os.path.join(root_model, '*.pkl'))
    for model_name in model_list:
        print('loading ' + model_name)

        net = senet154(pretrained=None)
        net.avg_pool = nn.AdaptiveAvgPool2d(1)
        net.last_linear = nn.Linear(net.last_linear.in_features, 2)

        net.load_state_dict(torch.load(model_name))
        net.cuda().eval()

        model_id = os.path.splitext(os.path.basename(model_name))[0]
        result_file = open(os.path.join(root_result, 'result_{}.txt'.format(os.path.basename(model_id))), 'w', 1)

        scores = dict()
        print('test start!')
        for i in trange(_dset.num_face):
            inp = torch.zeros(cfg.batch_size, 3, 128, 128)
            face = _dset._default_get(i)
            for j in range(cfg.batch_size):
                _face = np.copy(face)
                _face = _dset._augment(_face, i)
                inp[j] = _dset._to_tensor(_face)

            pred = net(inp.cuda())
            score = nn.functional.softmax(pred, 1)[:,1].mean().item()

            imgname = _dset.imdb.imgpath[i]
            imgname = os.path.basename(imgname).split('.')[0]
            result_file.write('{},{:.4f}\n'.format(imgname, score))
            scores[imgname] = score
            
        result_file.close()
        result.append(scores)
    
    return result

def get_average(root_result):
    result_files = glob(os.path.join(root_result, '*.txt'))

    results = list()
    for result_filename in result_files:
        with open(result_filename, 'r') as f:
            lines = f.readlines()

        scores = dict()
        for line in lines:
            imgname, score = line.split(',')
            scores[imgname] = float(score)
        
        results.append(scores)

    for result in results:
        assert result.keys() == results[0].keys()

    avg_scores = dict()
    for imgname in results[0].keys():
        avg_scores[imgname] = 0
    
    for result in results:
        for imgname in result.keys():
            avg_scores[imgname] += result[imgname]
    
    for imgname, score in avg_scores.items():
        avg_scores[imgname] = score / float(len(results))

    with open(os.path.join(root_result, 'avg.txt'), 'w') as f:
        for imgname, score in avg_scores.items():
            f.write('{},{:.4f}\n'.format(imgname, score))

    return avg_scores


if __name__ == '__main__':

    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parser = argparse.ArgumentParser(description='Test script')
    parser.add_argument('--tag', default='', type=str)
    parser.add_argument('--preset', default='gan', type=str)
    parser.add_argument('--root_dset', type=str, default='../dataset/test', help='Location of test set')
    parser.add_argument('--root_model', type=str, default='../pretrained/test', help='Location of models')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--avg', action='store_true')
    parser.add_argument('--set', default=None, type=str, nargs=argparse.REMAINDER, help='Optional settings')
    args = parser.parse_args()

    assert args.inference or args.avg
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    args.tag = now if len(args.tag) == 0 else args.tag

    cfg = get_config(args, now, os.getcwd())
    logger = Logger(cfg, '/tmp')

    cfg.reset = True
    cfg.clear_cache = True
    cfg.root_dset = args.root_dset
    cfg.root_model = args.root_model
    cfg.batch_size = args.batch_size

    root_result = os.path.abspath('../test')
    root_result = os.path.join(root_result, args.tag)
    shutil.rmtree(root_result, ignore_errors=True)
    os.makedirs(root_result, exist_ok=True)

    tic = time.time()

    if args.inference:
        result = test(cfg, logger, root_result)
    
    if args.avg:
        avg_scores = get_average(root_result)

    print('Time elapsed: {:.2f}'.format(time.time() - tic))
