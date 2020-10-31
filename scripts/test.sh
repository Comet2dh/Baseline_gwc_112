#!/usr/bin/env bash
set -x
DATAPATH="/media/data1/dh/DataSet/SceneFlowData/"
CUDA_VISIBLE_DEVICES='2,3'
python main.py --dataset sceneflow --cuda $CUDA_VISIBLE_DEVICES\
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 10 --batch_size 2 --lrepochs "5,8,10:2" \
    --model gwcnet-g --logdir ./checkpoints/PAM_1