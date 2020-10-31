#!/usr/bin/env bash
set -x
DATAPATH="/media/data1/dh/DataSet/SceneFlowData/KITTI2015"
python save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model gwcnet-g --loadckpt ./pre_checkpoints/kitti15/gwcnet-g/best.ckpt
