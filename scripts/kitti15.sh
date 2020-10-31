#!/usr/bin/env bash
set -x
DATAPATH="/media/data1/dh/DataSet/SceneFlowData/KITTI2015"
CUDA_VISIBLE_DEVICES='2, 3'
python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    --epochs 300 --lrepochs "200:10" \
    --model gwcnet-g --logdir ./checkpoints/2020_10_26/ocr_v1/kitti15 --loadckpt ./checkpoints//2020_10_26/ocr_v1/checkpoint_000015.ckpt \
    --batch_size 6 --test_batch_size 6