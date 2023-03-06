import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

LR_RATE = 0.0001


# class MSCM(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(MSCM, self).__init__()
#         self.convert = nn.Conv2d(in_channel, out_channel, 1)
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, 1), nn.ReLU(True),
#             nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
#             nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2), nn.ReLU(True),
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1), nn.ReLU(True),
#             nn.Conv2d(out_channel, out_channel, 3, padding=4, dilation=4), nn.ReLU(True),
#         )
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1), nn.ReLU(True),
#             nn.Conv2d(out_channel, out_channel, 3, padding=6, dilation=6), nn.ReLU(True),
#         )
#         self.score = nn.Conv2d(out_channel*4, 1, 3, padding=1)
#         # self.score = nn.Conv2d(out_channel*4, 1, kernel_size=3)

#     def forward(self, x):
#         x = self.convert(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#         x4 = self.branch4(x)

#         x = torch.cat((x1, x2, x3, x4), 1)
#         x = self.score(x)

#         return x




class RAS_mscm(nn.Module):
    def __init__(self):
        super(RAS_mscm, self).__init__()

        #MSCM
        #self.mscm = MSCM(512, 64)



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
        
        #侧输出6 global saliency的处理    
        self.conv1_dsn6 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2_dsn6 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.conv3_dsn6 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.conv4_dsn6 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.conv5_dsn6 = nn.Conv2d(256, 1, kernel_size=1)
        self.conv5_dsn6_up = nn.ConvTranspose2d(1, 1, kernel_size=64, stride=32) #global saliency up è‡? 
#         self.conv5_dsn6_4 = nn.ConvTranspose2d(1,1,kernel_size=8,stride=4) #global saliency up è‡?4
#         self.conv5_dsn6_3 = nn.ConvTranspose2d(1,1,kernel_size=16,stride=8)#global saliency ->3
#         self.conv5_dsn6_2 = nn.ConvTranspose2d(1,1,kernel_size=32,stride=16)#global saliency ->2
        
        #侧输出5
        self.conv5_dsn6_5 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.conv1_dsn5 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv2_dsn5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sum_dsn5_up = nn.ConvTranspose2d(1, 1, kernel_size=32, stride=16) # 5->1       
        self.conv4_dsn5_4 = nn.ConvTranspose2d(1,1,kernel_size=4,stride=2) # 5->4
        self.conv4_dsn5_3 = nn.ConvTranspose2d(1,1,kernel_size=8,stride=4) # 5->3
        self.conv4_dsn5_2 = nn.ConvTranspose2d(1,1,kernel_size=16,stride=8) # 5->2
        
        #侧输出4
        self.sum_dsn5_4 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.conv1_dsn4 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv2_dsn4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sum_dsn4_up = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=8) #4->1
        self.conv4_dsn4_3 = nn.ConvTranspose2d(1,1,kernel_size=4,stride=2) #4->3
        self.conv4_dsn4_2 = nn.ConvTranspose2d(1,1,kernel_size=8,stride=4) #4->2

        #侧输出3
        self.sum_dsn4_3 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.conv1_dsn3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv2_dsn3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sum_dsn3_up = nn.ConvTranspose2d(1, 1, kernel_size=8, stride=4) #3->1
        self.conv4_dsn3_2 = nn.ConvTranspose2d(1,1,kernel_size=4,stride=2)  #3->2

        #侧输出2
        self.sum_dsn3_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.conv1_dsn2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv2_dsn2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sum_dsn2_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2) #2->1

        #侧输出1
        self.conv1_dsn1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2_dsn1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        #添加短连接后->channel=1
        self.conv_dsn4 = nn.Conv2d(2, 1, kernel_size=1)
        self.conv_dsn3 = nn.Conv2d(2, 1, kernel_size=1)
        self.conv_dsn2 = nn.Conv2d(2, 1, kernel_size=1)
        self.conv_dsn1 = nn.Conv2d(2, 1, kernel_size=1)
        self.conv_fuse = nn.Conv2d(5, 1, kernel_size=1)

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

        #侧输出6 global saliency
        conv5_dsn6 = self.mscm(x)   #global saliency 经过mscm后 
        

        # x = self.conv1_dsn6(x)
        # x = F.relu(self.conv2_dsn6(x))
        # x = F.relu(self.conv3_dsn6(x))
        # x = F.relu(self.conv4_dsn6(x))
        # conv5_dsn6 = self.conv5_dsn6(x)  #global saliency 经过Conv后       

        #侧输出5
        x = self.conv5_dsn6_5(conv5_dsn6) #global saliency 放置6_5反卷积层
        crop1_dsn5 = self.crop(x, conv5_3.size())   #按照conv5_3的size crop global saliency
        x = -1*(torch.sigmoid(crop1_dsn5))+1   #1-sigmoid()
        x = x.expand(-1, 512, -1, -1).mul(conv5_3)  #残差模块  global saliency 与第五个侧输出相乘
        x = self.conv1_dsn5(x)
        x = F.relu(self.conv2_dsn5(x))
        x = F.relu(self.conv3_dsn5(x))   
        Feasure5 = self.conv4_dsn5(x)  #残差模块加权后的conv层 得Feasure5 

        #侧输出4
        x = self.sum_dsn5_4(Feasure5)  #上一个侧输出相加结果的上采样过程
        crop1_dsn4 = self.crop(x, conv4_3.size()) #将上一期预测按Conv4_3的size crop
        x = -1*(torch.sigmoid(crop1_dsn4))+1  #1-sigmoid()
        x = x.expand(-1, 512, -1, -1).mul(conv4_3) #残差模块的相乘
        x = self.conv1_dsn4(x)
        x = F.relu(self.conv2_dsn4(x))
        x = F.relu(self.conv3_dsn4(x))
        Feasure4 = self.conv4_dsn4(x)  #相乘后的Conv层 得Feasure4

        #侧输出3
        x = self.sum_dsn4_3(Feasure4)  
        crop1_dsn3 = self.crop(x, conv3_3.size())
        x = -1*(torch.sigmoid(crop1_dsn3))+1
        x = x.expand(-1, 256, -1, -1).mul(conv3_3)
        x = self.conv1_dsn3(x)
        x = F.relu(self.conv2_dsn3(x))
        x = F.relu(self.conv3_dsn3(x))
        Feasure3 = self.conv4_dsn3(x) #相乘后的Conv层 得Feasure3

        #侧输出2
        x = self.sum_dsn3_2(Feasure3)
        crop1_dsn2 = self.crop(x, conv2_2.size())
        x = -1*(torch.sigmoid(crop1_dsn2))+1
        x = x.expand(-1, 128, -1, -1).mul(conv2_2)
        x = self.conv1_dsn2(x)
        x = F.relu(self.conv2_dsn3(x))
        x = F.relu(self.conv2_dsn3(x))
        Feasure2 = self.conv4_dsn2(x) #相乘后的Conv层 得Feasure2

        #侧输出1
        x = self.sum_dsn2_up(Feasure2)
        crop1_dsn1 = self.crop(x,conv1_2.size()) #feasure2 crop到conv1_2
        x = -1*(torch.sigmoid(crop1_dsn1))+1
        x = x.expand(-1, 64, -1, -1).mul(conv1_2)
        x = self.conv1_dsn1(x)
        x = F.relu(self.conv2_dsn1(x))
        x = F.relu(self.conv3_dsn1(x))
        Feasure1 = self.conv4_dsn1(x) #相乘后的Conv层 得Feasure1

    
        #print
        print('Feasure1:')
        print(Feasure1.size())
        print('Feasure2:')
        print(Feasure2.size())
        print('Feasure3:')
        print(Feasure3.size())
        print('Feasure4:')
        print(Feasure4.size())
        print('Feasure5:')
        print(Feasure5.size())
        print('conv5_dsn6:')
        print(conv5_dsn6.size())
#         Feasure5_4 = self.crop(self.conv4_dsn5_4(Feasure5),Feasure4.size()) 
#         Feasure5_3 = self.crop(self.conv4_dsn5_3(Feasure5),Feasure3.size())
#         Feasure4_3 = self.crop(self.conv4_dsn4_3(Feasure4),Feasure3.size())
#         Feasure5_2 = self.crop(self.conv4_dsn5_2(Feasure5),Feasure2.size())
#         Feasure4_2 = self.crop(self.conv4_dsn4_2(Feasure4),Feasure2.size())
#         Feasure3_2 = self.crop(self.conv4_dsn3_2(Feasure3),Feasure2.size())
#         Feasure5_1 = self.crop(self.sum_dsn5_up(Feasure5),Feasure1.size())
#         Feasure3_1 = self.crop(self.sum_dsn3_up(Feasure3),Feasure1.size())
#         Feasure4_1 = self.crop(self.sum_dsn4_up(Feasure4),Feasure1.size())
#         Feasure2_1 = self.crop(self.sum_dsn2_up(Feasure2),Feasure1.size())
#         print('Feasure5_4:')
#         print(Feasure5_4.size())
#         print('Feasure5_3:')
#         print(Feasure5_3.size())
#         print('Feasure4_3:')
#         print(Feasure4_3.size())
#         print('Feasure5_2:')
#         print(Feasure5_2.size())
#         print('Feasure4_2:')
#         print(Feasure4_2.size())
#         print('Feasure3_2:')
#         print(Feasure3_2.size())
#         print('Feasure5_1:')
#         print(Feasure5_1.size())
#         print('Feasure4_1:')
#         print(Feasure4_1.size())
#         print('Feasure3_1:')
#         print(Feasure3_1.size())
#         print('Feasure2_1:')
#         print(Feasure2_1.size())
    
        # 添加短连接
        # 5 up-> x_size
        dsn5 = self.sum_dsn5_up(Feasure5) #5->1 
        dsn5_up = self.crop(dsn5, x_size) # up->x_size
        
        
        # 5->4   
        Feasure5_4 = self.crop(self.conv4_dsn5_4(Feasure5),Feasure4.size())      
        concat_dsn4 = torch.cat((Feasure5_4, Feasure4),1)
        score_dsn4 = self.conv_dsn4(concat_dsn4) # kernel = 1 
        score_dsn4_up = self.crop(self.sum_dsn4_up(score_dsn4), x_size) # up ->x_size 用于监督
        # score_dsn4_up = F.interpolate(score_dsn4, x_size, mode='bilinear', align_corners=True)
        
         #4->3 连接
        Feasure4_3 = self.crop(self.conv4_dsn4_3(Feasure4),Feasure3.size())
        concat_dsn3 = torch.cat((Feasure4_3, Feasure3),1)
        score_dsn3 = self.conv_dsn3(concat_dsn3) # kernel = 1
        score_dsn3_up = self.crop(self.sum_dsn3_up(score_dsn3),x_size)  # up 3->1 ->x_size 
        # score_dsn3_up  = F.interpolate(score_dsn3, x_size, mode='bilinear', align_corners=True)
        
        # 3->2 连接
        Feasure3_2 = self.crop(self.conv4_dsn3_2(Feasure3),Feasure2.size())
        concat_dsn2 = torch.cat((Feasure3_2,Feasure2),1)
        score_dsn2 = self.conv_dsn2(concat_dsn2) # kernel = 1
        score_dsn2_up = self.crop(self.sum_dsn2_up(score_dsn2),x_size)  # up ->x_size 
        # score_dsn2_up = F.interpolate(score_dsn2, x_size, mode='bilinear', align_corners=True)
       
         #  2->1 连接
        Feasure2_1 = self.crop(self.sum_dsn2_up(Feasure2),Feasure1.size())
        concat_dsn1 = torch.cat((Feasure2_1,Feasure1),1)
        score_dsn1 = self.conv_dsn1(concat_dsn1) # kernel = 1
        score_dsn1_up = self.crop(score_dsn1,x_size)  # up ->x_size 
        # score_dsn1_up = F.interpolate(score_dsn2, x_size, mode='bilinear', align_corners=True)
        
        #融合
        concat_upscore = torch.cat((dsn5_up, score_dsn4_up, score_dsn3_up,score_dsn2_up,score_dsn1_up),1)
        upscore_fuse = self.conv_fuse(concat_upscore)#连接后的特征 channel ->1
        

        def get_im(layer): 
            return torch.sigmoid(layer.clone()).detach().cpu().numpy()[0]*255.

        def save_im(path, im): return cv2.imwrite(path, np.mean(
            im, axis=0).reshape(im.shape[1], im.shape[2], 1).astype(np.uint8))

        if im_path_pre:
            layers = [Feasure5, Feasure4, Feasure3, Feasure2, Feasure1, dsn5_up,
                      score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up]
            im_paths = ["Feasure5", "Feasure4", "Feasure3", "Feasure2", "Feasure1", "dsn5_up",
                        "score_dsn4_up", "score_dsn3_up", "score_dsn2_up", "score_dsn1_up"]
            for l, i in zip(layers, im_paths):
                im = get_im(l)
                save_im(im_path_pre+i+".jpg", im)

#         return torch.sigmoid(upscore_fuse), torch.sigmoid(score_dsn1_up), torch.sigmoid(score_dsn2_up), torch.sigmoid(score_dsn3_up), torch.sigmoid(score_dsn4_up), torch.sigmoid(dsn5_up)
        return upscore_fuse, score_dsn1_up, score_dsn2_up, score_dsn3_up, score_dsn4_up, dsn5_up
    
  
    
#     def train(self, batch_x, batch_y):
#         fuse, dsn1, dsn2, dsn3, dsn4, dsn5 = self.forward(batch_x)
#         loss = nn.MSELoss()
# #         loss = nn.CrossEntropyLoss()
#         loss_1 = loss(dsn1, batch_y)
#         loss_2 = loss(dsn2, batch_y)
#         loss_3 = loss(dsn3, batch_y)
#         loss_4 = loss(dsn4, batch_y)
#         loss_5 = loss(dsn5, batch_y)
#         loss_fuse = loss(fuse, batch_y)
#         total_loss = loss_fuse + loss_1 + loss_2 + loss_3 + loss_4 + loss_5  #total_loss
#         self.optimzier.zero_grad()
#         total_loss.backward()
#         self.optimzier.step()
#         return total_loss
    
    def scheduler(self):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimzier, step_size=150, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimzier, mode = 'min',verbose=True,)
        return scheduler
    
    
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