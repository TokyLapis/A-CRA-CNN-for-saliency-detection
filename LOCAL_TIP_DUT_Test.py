import os
from skimage import io, img_as_ubyte, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 

import numpy as np
from PIL import Image
import glob

from data_loader_format import BlurdectTestDataset

from torchvision.utils import save_image
from model import RAS_mscm
#from model import LOCAL_TIP_Net
import matplotlib.pyplot as plt

# --------- 1. get image path and name ---------
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
image_dir = '/Users/chenyuliang/Desktop/shi100-source/'
label_dir = '/Users/chenyuliang/Desktop/shi100-gt/'
model_dir = '/Users/chenyuliang/Desktop/complementaryRA_new/saved_models_cat4/LOCAL_TIP_300.pth'
saved_dir = '/Users/chenyuliang/Desktop/complementaryRA_new/test_data/DUT_OMRON/'
img_name_list = glob.glob(image_dir + '*')
lbl_name_list = []
for img_path in img_name_list:
    img_name = img_path.split("DUT500S-Testing/")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    lab_name = imidx + ".bmp"

    lbl_name_list.append(label_dir + lab_name)

# --------- 2. dataloader ---------
# 1. dataload
test_blurdect_dataset = BlurdectTestDataset(img_name_list=img_name_list, lbl_name_list=lbl_name_list)
test_blurdect_dataloader = DataLoader(test_blurdect_dataset, batch_size=1, shuffle=False)

# --------- 3. model define ---------
print("...load LOCAL_Net...")
net = RAS_mscm()

state_dict = torch.load(model_dir,map_location='cpu')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v

# load params
net.load_state_dict(new_state_dict)
if torch.cuda.is_available():
    net.cuda()
net.eval()

# --------- 4. inference for each image ---------
for i_test, data_test in enumerate(test_blurdect_dataloader):

    print("inferencing:", img_name_list[i_test].split("/")[-1])

    inputs_test = data_test[0]
    # inputs_test = inputs_test/255.0
    labels_test = data_test[1]

    inputs_test = inputs_test.type(torch.FloatTensor)
    labels_test = labels_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
        labels_test = Variable(labels_test.cuda())
    else:
        inputs_test = Variable(inputs_test)
        labels_test = Variable(labels_test)

    # out1, out2, out3, out4, out5,_,_,_,_,_ = net(inputs_test)
    out,_,_,_,_  = net(inputs_test)
    #out1 = (out1[:,0,:,:].unsqueeze(1)+1-out1[:,1,:,:].unsqueeze(1))/2.0
    out = out[:,0,:,:].unsqueeze(1)

    out = out.detach().repeat(1, 3, 1, 1)

    img_name_split = lbl_name_list[i_test].split("/")[-1]

    img_path1 = saved_dir + img_name_split


    labels_test = labels_test.repeat(1, 3, 1, 1)


    save_image(out.data.cpu(), img_path1, nrow=out.shape[0], padding=0)

    print('Process %s OK' % i_test)







