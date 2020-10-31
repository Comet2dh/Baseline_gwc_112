import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

def getCAM(cost, image):
	cost_np = cost.detach().cpu().numpy()
	img = F.interpolate(image, scale_factor=0.25)
	img_np = img.detach().cpu().numpy()
	img_np = np.transpose(img_np[0], (1, 2, 0))
	# d张概率图（这里d=D/4=48）
	batch, n_class, height, width = cost_np.shape
	# Normalize
	img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255

	# get heatmap
	heatmap = []
	out = []
	for i in range(n_class):
		cam_resize = cost_np[0, i, :, :]
		cam_resize = (cam_resize - cam_resize.min()) / (cam_resize.max() - cam_resize.min()) * 255
		pro = cam_resize.astype(np.uint8)
		pro_255 = np.expand_dims(pro, axis=2)
		heatmap.append(cv2.applyColorMap(pro_255, cv2.COLORMAP_JET))
		heatmap[i] = cv2.cvtColor(heatmap[i], cv2.COLOR_BGR2RGB).astype(np.float32)
		# zero out
		heatmap[i][np.where(pro <= 80)] = 0
		out.append(cv2.addWeighted(src1=img_np, alpha=0.5, src2=heatmap[i], beta=0.5, gamma=0))
		# out.append(img_np[0])
	cams = torch.Tensor(out)
	return cams.permute(0, 3, 1, 2)

# save CAM to savepth
# def saveCAM(imgUrl, savepath, pred, CAM):
# 	if not os.path.exists(savepath):
# 		os.mkdir(savepath)
#
# 	name = imgUrl.split('/')[-1]
# 	savename = name.split('.')[0] + '_' + str(pred)
#
# 	img = cv2.imread(imgUrl, 1)
#
# 	attentionmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
# 	overlap = attentionmap * 0.5 + img * 0.5
#
# 	cv2.imwrite(os.path.join(savepath, savename + '_GradCAM.png'), attentionmap)
# 	cv2.imwrite(os.path.join(savepath, savename + '_overlap.png'), overlap)