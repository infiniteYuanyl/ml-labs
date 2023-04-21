# coding=utf-8
import numpy as np
import struct
import os
import time

class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
    def forward(self, input):  # 前向传播计算
        start_time = time.time()
        self.input = input
        # TODO：全连接层的前向传播，计算输出结果
        # self.output = ________________
        self.output = np.dot(self.input,self.weight) + self.bias
        #input = d * num_input ,output = (d+1) * num_output
        return self.output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：全连接层的反向传播，计算参数梯度和本层损失
        # self.d_weight = ________________
        # self.d_bias = ________________
        # bottom_diff = ________________
        # 链式求导，y = wx + b，dw = dloss * x
        self.d_weight = np.dot(self.input.T,top_diff)
        #db = dloss * 1，但考虑到 dloss的形状为 num_output * d，因此要对行方向进行求和才能匹配b
        #其中input=d * num_input
        self.d_bias = np.sum(top_diff,axis=0)
        bottom_diff = np.dot(top_diff,self.weight.T)
        return bottom_diff
    def update_param(self, lr):  # 参数更新
        # TODO：对全连接层参数利用参数进行更新
        # self.weight = ________________
        # self.bias = ________________
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr *  self.d_bias
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self):  # 参数保存
        return self.weight, self.bias

class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input
        # TODO：ReLU层的前向传播，计算输出结果
        # output = ________________
        # relu,小于0置0，这样让那些> 0的位置保持不变，小于等于0的位置，self.input 置0，对应位置也被置0了。
        output = self.input * (self.input > 0)
        return output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：ReLU层的反向传播，计算本层损失
        # bottom_diff = ________________
        #同理，把那些input > 0的位置全部 * 1（relu的求导），原来被抑制为0的位置不变，不传播损失。
        bottom_diff = top_diff * ( self.input >0 )
        return bottom_diff

class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')
    def forward(self, input):  # 前向传播的计算
        # TODO：softmax 损失层的前向传播，计算输出结果
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        # self.prob = ________________

        # self.prob =  np.exp((input -input_max) - np.log(np.sum(input_exp,axis=1,keepdims=True).reshape(-1,1)) )
        self.prob = input_exp / np.sum(input_exp,axis=1,keepdims=True).reshape(-1,1)
        return self.prob
    def get_loss(self, label):   # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):  # 反向传播的计算
        # TODO：softmax 损失层的反向传播，计算本层损失
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        # prob_sum = np.sum(self.prob,axis=1,keepdims=True).reshape(-1,1)
        # prob_x_minus_max = self.prob - np.max(self.prob,axis=1,keepdims=True).reshape(-1,1)
        # prob_x2_minus_max = self.prob * 2 - np.max(self.prob,axis=1,keepdims=True).reshape(-1,1)
        # # softmax公式推导即可
        # # dloss = loss * (e^(x-max)*sum - e^(x-max) * e^x)/ (sum^2)
        # bottom_diff = loss * (((np.exp(prob_x_minus_max))/prob_sum) - np.exp(prob_x2_minus_max ) * (prob_sum **2))
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff
class DropoutLayer(object):
    def __init__(self,p=0.3):  
        self.p =p     
        print('\tDropout layer.')
    def forward(self, input):  # 前向传播的计算
        self.input = input
        np.random.seed(1234)
        self.mask = (np.random.rand(self.input.shape[0],self.input.shape[1]) >=self.p / (1 - self.p))    
        
        self.input = self.input * self.mask
        output = self.input
        return output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：Dropout层的反向传播，计算本层损失
        
        bottom_diff = top_diff * self.mask 
        return bottom_diff


    

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, tt=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if tt == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) / self.stride + 1
        width_out = (width - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw] =  np.sum(self.input_pad[idxn, :,idxh*self.stride:idxh*\
                            self.stride+self.kernel_size,idxw*self.stride:idxw*self.stride+self.kernel_size] * self.weight[:,:,:,idxc]) + self.bias[idxc]
        self.forward_time = time.time() - start_time
        return self.output
    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        # 把卷积核转换为行向量，对应局部数据hs：hs+k，ws：ws+k转化成列向量
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) / self.stride + 1
        width_out = (width - self.kernel_size) / self.stride + 1
        # cin,k,k,cout -> cin*k*k,cout
        self.weight_reshape = np.reshape(self.weight,(-1,self.channel_out))
        # img2col:N * H * W,cin*k*k
        self.img2col = np.zeros( [self.input.shape[0]*height_out*width_out ,self.channel_in*self.kernel_size*self.kernel_size ])
        
        # 将img转换为列向量
        for idxn in range(self.input.shape[0]):
            for idxh in range(height_out):
                for idxw in range(width_out):
                    #计算出转换后的行索引
                    row_i = idxn*height_out*width_out+idxh*width_out+idxw                
                    # img2:cin*k*k,input_pad:cin,k,k,reshape:cin,k,k->cin*k*k
                    self.img2col[row_i,:]= self.input_pad[idxn,:,idxh*self.stride:idxh*\
                            self.stride+self.kernel_size,idxw*self.stride:idxw*self.stride+self.kernel_size].reshape([-1])
        #output = img2col x weight :N * H * W ,cout
        output = np.dot(self.img2col ,self.weight_reshape) + self.bias
        self.output = output.reshape([self.input.shape[0],height_out,width_out,-1]).transpose([0,3,1,2])
        
        self.forward_time = time.time() - start_time
        return self.output
    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        # # padding
       
        # top_diff->top_diff_pad_reshape:N ,cout,H , W => N * H * W ,cout
        top_diff_pad = np.zeros([top_diff.shape[0], top_diff.shape[1], top_diff.shape[2]+2*self.kernel_size-2, top_diff.shape[3]+2*self.kernel_size-2])
        top_diff_pad[:, :, self.kernel_size-1:1-self.kernel_size, self.kernel_size-1:1-self.kernel_size] = top_diff
        top_diff_pad_reshape = np.zeros([self.input.shape[0]*self.input_pad.shape[2]*self.input_pad.shape[3], top_diff.shape[1]*(self.kernel_size**2)])
        for idxn in range(self.input_pad.shape[0]):
            for idxh in range(self.input_pad.shape[2]):
                for idxw in range(self.input_pad.shape[3]):
                    row_i = idxn * self.input_pad.shape[2] * self.input_pad.shape[3] + idxh * self.input_pad.shape[3] + idxw
                    top_diff_pad_reshape[row_i,:] = top_diff_pad[idxn, :,\
                         idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size].reshape([-1])   
        # bottom_diff: N , cin ,H ,W
        # top_diff_pad_reshape:N * H * W ,cout *k *k,
        # weight_reshape :cin*k*k,cout
        
        
        # bottom_diff = np.matmul(top_diff_pad_reshape , self.weight_reshape.reshape(self.channel_in,(self.kernel_size**2),-1).\
        #     transpose(1,2,0).reshape(-1,self.channel_in))
        bottom_diff = np.matmul(top_diff_pad_reshape , np.rot90(self.weight, k=2, axes=(1,2)).transpose(3,1,2,0).reshape(-1, self.channel_in))
        
        bottom_diff = np.reshape(bottom_diff,[self.input_pad.shape[0],self.input_pad.shape[2],self.input_pad.shape[3],self.channel_in]).transpose(0,3,1,2)
        
        bottom_diff = bottom_diff[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]]
        
        
        
        self.backward_time = time.time() - start_time
        
        return bottom_diff
    def backward_raw(self, top_diff):
        # print('top_diff:',top_diff.shape)
        # print('input',self.input.shape)
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        self.d_weight[:, :, :, idxc] += top_diff[idxn,idxc,idxh,idxw] * \
                            self.input_pad[idxn,:,idxh*self.stride: idxh*self.stride+ self.kernel_size,\
                                idxw*self.stride: idxw*self.stride+ self.kernel_size]
                        self.d_bias[idxc] += top_diff[idxn,idxc,idxh,idxw]
                        #3.5公式，x_pad 反向传播
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += \
                            top_diff[idxn,idxc,idxh,idxw] * self.weight[:,:,:,idxc]
        bottom_diff = bottom_diff[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]]
        self.backward_time = time.time() - start_time
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def get_forward_time(self):
        return self.forward_time
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, tt=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw_book
        if tt == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        self.output[idxn,idxc,idxh,idxw]= np.max(self.input[idxn,idxc,idxh* self.stride:idxh* \
                            self.stride+self.kernel_size,idxw* self.stride:idxw*self.stride+self.kernel_size])
                        #   最大池化层，前向传播需要记忆每个max的位置。使用argmax函数获取其索引。
                        curren_max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride\
                            +self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        # 然后需要求出数组某元素拉成一维后的索引值在原本维度（或指定新维度）中对应的索引(这里是相对原图像素idxh，idxw而言的)
                        # 使用np.unravel_index函数，恢复max元素的一维索引在原图上对应最大池化的位置，恢复形状为kernel_size^2
                        # 如 一个矩阵3*3，原kernel=[[1,2,3],[4,5,6],[7,8,0]],argmax后index=7,恢复后位，index转为[2,1]。
                        # 这里，current_max_index是一个 2 * 3的矩阵，第一维是h方向上，最大索引的列坐标；对w同理。
                       
                        curren_max_index = np.unravel_index(curren_max_index, [self.kernel_size, self.kernel_size])
                        #利用此索引，可以获得最大索引在卷积核的位置，在max_index中标记为1.
                        self.max_index[idxn, idxc, idxh*self.stride+curren_max_index[0], idxw*self.stride+curren_max_index[1]] = 1
        return self.output
    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        
        self.input = input
        height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        self.input_col = np.zeros([self.input.shape[0], self.input.shape[1], height_out*width_out, self.kernel_size**2])
        for i in range(height_out):
            for j in range(width_out):
                self.input_col[:,:,i*width_out+j,:] = self.input[:, :, i*self.stride:i*self.stride+self.kernel_size, j*\
                    self.stride:j*self.stride+self.kernel_size].reshape(self.input.shape[0], self.input.shape[1], -1)
        self.output_col = np.max(self.input_col, axis=3)
        max_index_col = np.zeros([self.input.shape[0]*self.input.shape[1]*height_out*width_out, self.kernel_size**2])
        max_index_col[np.arange(self.input.shape[0]*self.input.shape[1]*height_out*width_out), np.argmax(self.input_col, axis=3).reshape(-1)] = 1
        max_index_col = max_index_col.reshape([self.input.shape[0], self.input.shape[1], height_out*width_out, self.kernel_size**2])
        self.output = self.output_col.reshape([self.input.shape[0], self.input.shape[1], height_out, width_out])
        self.max_index = np.zeros(self.input.shape)
        for i in range(height_out):
            for j in range(width_out):
                self.max_index[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]\
                     = max_index_col[:,:,i*width_out+j,:].reshape([self.input.shape[0], self.input.shape[1], self.kernel_size, self.kernel_size])

        return self.output
    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速

        bottom_diff = np.multiply(self.max_index, top_diff.repeat(self.kernel_size,2).repeat(self.kernel_size,3))

        return bottom_diff
    def backward_raw_book(self, top_diff):
        #只有最大值索引才被传递损失，其他地方不传递
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO: 最大池化层的反向传播， 计算池化窗口中最大值位置， 并传递损失
                        # self.max_index是取一张图片一个通道维度的图中的最大值索引,并且置了1.
                        # 使用np.argwhere把非零元素的位置提取出来.
                        # 最后取0实际上是为了降维。
                        max_index = np.argwhere(self.max_index[idxn, idxc, \
                            idxh*self.stride:idxh*self.stride+self.kernel_size, \
                            idxw*self.stride:idxw*self.stride+self.kernel_size])[0]
                        bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] =\
                        top_diff[idxn,idxc,idxh,idxw]
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output
    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff

