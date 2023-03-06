import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

LR_RATE = 0.0001


class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.convs_cat = nn.Sequential(
            nn.Conv2d(out_channel*3 + 1, out_channel*2, 3, padding = 1),nn.ReLU(True),
            nn.Conv2d(out_channel*2, out_channel*2, 3, padding = 1),nn.ReLU(True),
            nn.Conv2d(out_channel*2, out_channel, 3, padding = 1),
            nn.Sigmoid()
        )
        self.channel = out_channel

    def forward(self, x, y):


        #-----------------concat sigmoid, 1- sigmoiud, 上一层mask, 当前层feature map--------------------
        a = torch.sigmoid(-y)   #feature
        a1 = torch.sigmoid(y)    #feature
        x = self.convert(x)   #feature
        product = a.expand(-1, self.channel, -1, -1).mul(x)
        product1 = a1.expand(-1, self.channel, -1, -1).mul(x)
        connect = torch.cat((product, product1, y, x),1)
        x = self.convs_cat(connect)
        x = self.convs(x)

        return x


class MSCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSCM, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
            nn.Sigmoid()
            
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2), nn.ReLU(True),
            nn.Sigmoid()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=4, dilation=4), nn.ReLU(True),
            nn.Sigmoid()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=6, dilation=6), nn.ReLU(True),
            nn.Sigmoid()
        )
        self.score = nn.Conv2d(out_channel*4, 1, 3, padding=1)
        # self.score = nn.Conv2d(out_channel*4, 1, kernel_size=3)

    def forward(self, x):
        #x = self.relu(self.bn(self.convert(x)))
        x = self.convert(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.score(x)

        return x

class RAS_mscm(nn.Module):
    def __init__(self,channel=64):
        super(RAS_mscm, self).__init__()

        
       
        #VGG 啦啦啦
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
       
  

        #MSCM
        self.mscm = MSCM(512, channel)

        #RA module
        self.ra5 = RA(512,channel)
        self.ra4 = RA(512,channel)
        self.ra3 = RA(256,channel)
        self.ra2 = RA(128,channel)
        self.ra1 = RA(64,channel)

      

        self.optimzier = optim.Adam(self.parameters(), lr=LR_RATE, weight_decay = 0.0005)
        
        #不用递归方式 初始化权重，用下面的for循环初始化
        
        #self.apply(RAS.weights_init)

        #加载pytorch的内置VGG16的模型  使用预训练参数
        vgg16 = models.vgg16(pretrained=True)
        weights = vgg16.state_dict()
        weights_index = ['features.0.weight','features.2.weight','features.5.weight','features.7.weight','features.10.weight','features.12.weight','features.14.weight','features.17.weight','features.19.weight','features.21.weight','features.24.weight','features.26.weight','features.28.weight']
        
        #index是VGG的一共13个网络层的索引，超出这个范围按照以前的方式初始化
        index = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if index <13:
                    m.weight.data = weights[weights_index[index]]
                    index += 1
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
   
        
    
    def forward(self, x, im_path_pre=False):

        x_size = x.size()
       

        ##VGG16的5个卷积块
        x = F.relu(self.conv1_1(x))
        conv1_2 = F.relu(self.conv1_2(x))
        x = F.max_pool2d(conv1_2, kernel_size=2, stride=2)

        x = F.relu(self.conv2_1(x))
        conv2_2 = F.relu(self.conv2_2(x))
        x = F.max_pool2d(conv2_2, kernel_size=2, stride=2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        conv3_3 = F.relu(self.conv3_3(x))
        x = F.max_pool2d(conv3_3, kernel_size=2, stride=2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        conv4_3 = F.relu(self.conv4_3(x))
        x = F.max_pool2d(conv4_3, kernel_size=3, stride=2, padding=1)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        conv5_3 = F.relu(self.conv5_3(x))
        x = F.max_pool2d(conv5_3, kernel_size=3, stride=2, padding=1)


        #-------------------------------mscm module---------------------------------------------
        #侧输出6 global saliency
        conv5_dsn6 = self.mscm(x)   #global saliency 经过mscm后 
        




        #----------------------------------------RA module ----------------------------------------------

        #侧输出5 RAmodule
        x = F.interpolate(conv5_dsn6, conv5_3.size()[2:], mode='bilinear', align_corners=True)
        Feasure5 = self.ra5(conv5_3, x)


        #侧输出4 RAmodule
        x = F.interpolate(Feasure5, conv4_3.size()[2:], mode='bilinear', align_corners=True)
        Feasure4 = self.ra4(conv4_3, x)

        #侧输出3 RAmodule
        x = F.interpolate(Feasure4, conv3_3.size()[2:], mode='bilinear', align_corners=True)
        Feasure3 = self.ra3(conv3_3, x)

        #侧输出2 RAmodule
        x = F.interpolate(Feasure3, conv2_2.size()[2:], mode='bilinear', align_corners=True)
        Feasure2 = self.ra2(conv2_2, x)

        #侧输出1 RAmodule
        x = F.interpolate(Feasure2, conv1_2.size()[2:], mode='bilinear', align_corners=True)
        Feasure1 = self.ra1(conv1_2, x)
        #--------------------------------------RA module  End ----------------------------------------

    
        # 添加短连接
        # 5 up-> x_size
        dsn5_up = F.interpolate(Feasure5, x_size[2:], mode='bilinear', align_corners=True) #5->1 
        
        
        # # 5->4   
        # Feasure5_4 = F.interpolate(Feasure5, Feasure4.size()[2:], mode='bilinear', align_corners=True)     #feature5.size() --> feature
        # concat_dsn4 = torch.cat((Feasure5_4, Feasure4),1)   #feature5 feature4 concat
        # score_dsn4 = self.conv_dsn4(concat_dsn4) # kernel = 1 channel = 1
        score_dsn4_up = F.interpolate(Feasure4, x_size[2:], mode='bilinear', align_corners=True)   #feature --> up
        
        #  #4->3 连接
        # Feasure4_3 = F.interpolate(Feasure4, Feasure3.size()[2:], mode='bilinear', align_corners=True) 
        # concat_dsn3 = torch.cat((Feasure4_3, Feasure3),1)
        # score_dsn3 = self.conv_dsn3(concat_dsn3) # kernel = 1
        score_dsn3_up  = F.interpolate(Feasure3, x_size[2:], mode='bilinear', align_corners=True)
        
        # # 3->2 连接
        # Feasure3_2 = F.interpolate(Feasure3, Feasure2.size()[2:], mode='bilinear', align_corners=True) 
        # concat_dsn2 = torch.cat((Feasure3_2,Feasure2),1)
        # score_dsn2 = self.conv_dsn2(concat_dsn2) # kernel = 1
        score_dsn2_up = F.interpolate(Feasure2, x_size[2:], mode='bilinear', align_corners=True)
       
        #  #  2->1 连接
        # Feasure2_1 = F.interpolate(Feasure2, Feasure1.size()[2:], mode='bilinear', align_corners=True) 
        # concat_dsn1 = torch.cat((Feasure2_1,Feasure1),1)
        # score_dsn1 = self.conv_dsn1(concat_dsn1) # kernel = 1
        score_dsn1_up = F.interpolate(Feasure1, x_size[2:], mode='bilinear', align_corners=True)
        
        #融合
        # concat_upscore = torch.cat((dsn5_up, score_dsn4_up, score_dsn3_up,score_dsn2_up,score_dsn1_up),1)
        # upscore_fuse = self.conv_fuse(concat_upscore)#连接后的特征 channel ->1
        

        # def get_im(layer): 
        #     return torch.sigmoid(layer.clone()).detach().cpu().numpy()[0]*255.

        # def save_im(path, im): return cv2.imwrite(path, np.mean(
        #     im, axis=0).reshape(im.shape[1], im.shape[2], 1).astype(np.uint8))

        # if im_path_pre:
        #     layers = [Feasure5, Feasure4, Feasure3, Feasure2, Feasure1, dsn5_up,
        #               score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up]
        #     im_paths = ["Feasure5", "Feasure4", "Feasure3", "Feasure2", "Feasure1", "dsn5_up",
        #                 "score_dsn4_up", "score_dsn3_up", "score_dsn2_up", "score_dsn1_up"]
        #     for l, i in zip(layers, im_paths):
        #         im = get_im(l)
        #         save_im(im_path_pre+i+".jpg", im)

        # return torch.sigmoid(upscore_fuse), torch.sigmoid(score_dsn1_up), torch.sigmoid(score_dsn2_up), torch.sigmoid(score_dsn3_up), torch.sigmoid(score_dsn4_up), torch.sigmoid(dsn5_up)
        return score_dsn1_up, score_dsn2_up, score_dsn3_up, score_dsn4_up, dsn5_up
    
  
    
#     def train(sel
    # def scheduler(self):
    #     scheduler = torch.optim.lr_scheduler.StepLR(self.optimzier, step_size=150, gamma=0.5)
    #     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimzier, mode = 'min',verbose=True,)
    #     return scheduler
    
    
    # def test(self, batch_x, im_path_pre=False):
    #     if im_path_pre:
    #         assert(batch_x.size()[0] == 1)
    #     dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.forward(batch_x, im_path_pre)

    #     #forward函数的返回值为：upscore_fuse, score_dsn1_up, score_dsn2_up, score_dsn3_up, score_dsn4_up, dsn5_up
    #     #这里应该只返回 最终的fuse后的结果就行了 
    #     return dsn1.detach()

    #     # dsn = dsn1.detach()+dsn2.detach()+dsn3.detach() + \
    #     #     dsn4.detach()+dsn5.detach()+dsn6.detach()
    #     # return dsn/6

    def crop(self, upsampled, x_size):
        c = (upsampled.size()[2] - x_size[2]) // 2
        _c = x_size[2] - upsampled.size()[2] + c
        assert(c >= 0)
        if(c == _c == 0):
            return upsampled
        return upsampled[:, :, c:_c, c:_c]
'''
    @staticmethod
    #m 权重文件  
    def weights_init(m):
        classname = m.__class__.__name__
        print(m)
        if isinstance(m, nn.Conv2d):
            aa = m.weight
            print(aa.shape)
            pass
'''            
            #print(m.weight.data.shape)
            #m.weight.data = torch.ones(m.weight.data.shape)
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#         elif isinstance(m, nn.ConvTranspose2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))