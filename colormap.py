from matplotlib import pyplot as plt
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import os
from PIL import Image
import matplotlib.image as mpimg

os.makedirs('./groundtruth', exist_ok=True)
datapath='/media/data1/dh/DataSet/SceneFlowData/KITTI2015/testing/disparity'

disparity_dir = os.listdir(datapath)
for d in disparity_dir:
    fn = os.path.join("groundtruth", d)
    dd = os.path.join(datapath, d)
    disp_groundtruth = mpimg.imread(dd)
    print("saving to", fn)

    plt.imshow(disp_groundtruth, cmap="hsv")
    plt.savefig(fn)
