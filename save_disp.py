from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
from skimage import io
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-init', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/media/data1/dh/DataSet/SceneFlowData', help='data path')
parser.add_argument('--testlist', default='./filenames/sceneflow_test.txt', help='testing list')
parser.add_argument('--loadckpt', default='./checkpoints/2020_10_21/baseline/checkpoint_000015.ckpt', help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


def test():
    os.makedirs('./predictions', exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        disp_est, errormap = test_sample(sample)
        disp_est_np = tensor2numpy(disp_est)
        errormap_np = tensor2numpy(errormap)
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))

        for disp_est, errormap, top_pad, right_pad, fn in zip(disp_est_np, errormap_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            fn_disp = os.path.join("predictions", fn.split('/')[-1])
            fn_err = os.path.join("predictions", 'err_'+fn.split('/')[-1])
            print("saving to", fn_disp, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            io.imsave(fn_err, errormap)
            io.imsave(fn_disp, disp_est_uint)
            #plt.imshow(disp_est_uint, cmap="hsv")
            #plt.savefig(fn)

# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_ests = model(imgL, imgR)
    errormap = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
    return disp_ests[-1], errormap[-1]


if __name__ == '__main__':
    test()
