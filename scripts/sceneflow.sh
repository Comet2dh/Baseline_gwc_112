#!/usr/bin/env bash
set -x
DATAPATH="/media/data1/dh/DataSet/SceneFlowData/"
python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 16 --lrepochs "10,12,14,16:2" \
    --model gwcnet-g --logdir ./checkpoints/sceneflow/gwcnet-g/2020_10_6_dispAttention