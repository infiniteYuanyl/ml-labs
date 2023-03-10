# 2023年春季《机器学习》实验报告

## 实验一 多元线性回归

| 姓名           | 袁钰林                                                   |
| -------------- | -------------------------------------------------------- |
| **学号**       | **20030021092**                                          |
| 专业班级       | 20计算机科学与技术1班                                    |
| 课程名称       | 中国海洋大学23春《机器学习》                             |
| 任课老师       | 李岳尊                                                   |
| 完成日期       | 2023.2.27                                                |
| 实验名称       | 多元线性回归                                             |
| Github源码地址 | https://github.com/infiniteYuanyl/ml-labs/tree/lab1/labs |



## **一、实验目标**

**对波士顿房价进行预测**

<img src="https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230227131527764.png" alt="image-20230227131527764" style="zoom: 67%;" />

**1.将数据集拆分成训练集（前450个样本）和测试集（后56个样本）**
**2.利用多元线性回归模型（最小二乘法或者梯度下降法均可）对训练数据进行拟合**
**3.对拟合得到的模型在测试集上进行测试，使用均方误差作为实验的准确结果并输出。**



## 二、实验内容

#### **1.拆分数据集和数据归一化**

```python
import pandas as pd
import numpy as np

def read_txt(args):
    meta = np.genfromtxt(args.data,dtype='float64')

    data = meta
    features_num = data.shape[1]
    for idx in range(features_num-1):
        #最后一维是标签，不用归一化
        data[:, idx] = (data[:, idx] - np.mean(data[:, idx])) / np.std(data[:, idx])  		  # 列均值方差归一化
    train_index = int(args.split)

    train_data = data[:train_index,:]
    test_data = data[train_index:,:]

    print('train_data {} * {}'.format(train_data.shape[0], train_data.shape[1]))

    print('test_data {} * {}'.format(test_data.shape[0], test_data.shape[1]))


    return train_data,test_data
```

通过命令行键入 `train_index`来确认哪些是验证集哪些作为验证集。由于实验要求前450作为训练集，因此这里不用k折交叉验证。

**特征归一化**是为了让每个参数的尺度统一，在更新参数的时候不需要考虑参数尺度的因素去调整。如下图所示，如果不归一化，在同一个学习率下，尺度更大的参数w1需要迈更大的步子才能和参数W2持平，不利于我们调参（说白了，如果不归一化，我们要为每个参数设置专门的步长）。归一化有助于我们选择统一的步长（学习率），有利于我们模型的收敛。

![image-20230310230006303](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimgimage-20230310230006303.png)

#### **2.使用梯度下降法进行参数拟合**

```python
 def step_by_gradient(self,dloss,x):
        #print(self.lr * np.dot(x,dloss))
        self.params = self.params - self.lr * np.dot(x,dloss)
        
 def forward(self,x,idx,labels,eval=False):
        #predict 为一个1 * 1 的数
        predict = np.dot(x.T,self.params)
        # print('predict',predict.shape)
        loss,dloss = self.MSEloss(predict,labels[idx])
        if not eval :
            self.update_params(dloss,x)
        return 0
```

核心代码如上，梯度更新公式将在实验原理给出。代码有一定冗余，不过方便打印一些变量。

#### **3.训练**

```Python
    def train(self,epochs):
        num = self.train_data.shape[0]
        print('train_num :', num)
        for epoch in range(int(epochs)):

            loss_sum =0.0
            data,labels = self.split_data(self.train_data)
            for index in range(num):
                x = data[index,:]

                x = np.append(x,1)
                # print(x.shape)
                self.forward(x,index,labels)
                loss_sum+=float(self.loss)
            if epoch % 100 ==1:print('epoch: {} MSEloss: {}'.format(epoch+1, 2 * loss_sum/num))
            if self.lr >=0.0001:self.lr = self.lr * self.beta
            np.random.shuffle(self.train_data)
            if epoch % 100 ==1:print(self.params)
        print('Train over')
        d= datetime.today().strftime('%Y%m%d_%H_%M_%S')
        np.savez('checkpoints/'+d+'_paramsW.npz',W=self.params,LR=self.lr)
```

训练函数如上，可以通过命令行输入训练的epochs数量，同时每隔100epoch打印一次当前轮MSE的loss信息。同时学习率也可以动态下降，低于0.0001时保持不变。这里由于没必要（怕麻烦）就不手搓SGD了。

**最后模型训练结束后可以保存可学习参数和学习率，可供机器继续学习。**

#### **4.测试**

```Python
def test(self):
    print('test start')
    loss_sum =0.0
    print('test cases:',self.test_data.shape[0])
    if self.test_data is None:
        raise ValueError('test_data not loaded!')
    num = self.test_data.shape[0]
    self.eval=True
    data, labels = self.split_data(self.test_data)
    for index in range(num):
        x = data[index,:]
        x = np.append(x, 1)
        self.forward(x,index,labels,eval=True)
        print('item {} : loss: {}'.format(index + 1, self.loss))
        loss_sum +=self.loss

    print('Test over! MSTloss:{},mAcc: {}'.format( loss_sum / num,self.Acc/num))
```

测试函数如上，由于是测试所有数据集不选择shuffle打乱。需要注意的是，在测试的时候我们不需要更新参数，于是设置eval=1（表示评估状态），不更新参数。训练5k epochs后，测试1/2 的MSEloss=5.64，MSEloss为5.64 * 2 = 11.28.这里使用1/2的原因是后续原理会给出。

最终运行结果。这里使用的loss是 1/2 的MSEloss

![image-20230227223033829](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimgimage-20230227223033829.png)



## 三、实验原理

#### **1.梯度下降法**

```python
self.params = self.params - self.lr * np.dot(x,dloss)
```



![image-20230227224949697](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimgimage-20230227224949697.png)

核心公式如上，J为对参数的偏导数，这里就是x × dloss了，α是学习率。由于x就是一个一维向量，因此也不必转置。



#### **2.线性回归**

**Y=Wx+b**

这里的W是一个具有`13`维特征的向量，`x`也是13维的向量，`b`是一个常数。

为了计算，这里将`b`融入`W`可学习参数矩阵中，相当于给`W`增加一个维度，默认值为`b`；因此，`x`也需要增加一个维度，默认为1.这样，`Y=W'x'`了。

```python
    def initialize_weight(self):
        # +1 为b的那一项，同时label也需要经过处理
        if self.load_checkpoint is not None:
            with np.load(self.load_checkpoint) as f:
                self.params = f['W']
                self.lr =f['LR']

            print('load checkpoint from {}'.format(self.load_checkpoint))
        else:
            self.params = 1.0 * np.ones(self.feature_num+1,dtype=np.float64)
            print('initializing weight！')
        print('weight:',self.params.shape)
        print('features_num:',self.feature_num)
```

在初始化权重矩阵中可以看出，这里我默认`b`设置为了1.而**训练**的数据输入x也做了相关处理，增加了一维，值为1。下面是书上公式对x的扩充。

![image-20230227230134289](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimgimage-20230227230134289.png)



#### 3.数据打乱&命令行参数设置

为了不让训练的数据学习到输入数据的顺序特征，在训练时，编写了`split_data.py`设置shuffle参数为True可以对数据和标签进行随机打乱重排，为了就是增加泛化程度并抑制一定过拟合。

```python
    def split_data(self,origin_data,shuffled=False):
        if shuffled:
            np.random.shuffle(origin_data)
        data = origin_data[:,:-1]
        labels = origin_data[:,-1]
        return data,labels
```

为了优雅的调试，浅浅做了一个命令行。

参数:数据路径`data`,分割索引`split`,参数更新方式`update`,最大训练轮数`epochs`,检查点载入路径`ckpg`

```python
import argparse
from linear_regression import LinearRegression
from  data_preprocess import read_txt
def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--split', help='split index')
    parser.add_argument('--update', help='update type')
    parser.add_argument('--epochs', help='max epochs to train')
    parser.add_argument('--ckpg', help='max epochs to train')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_data,test_data = read_txt(args)
    if args.ckpg is None:
        net = LinearRegression(train_data, test_data)
    else:
        net = LinearRegression(train_data, test_data,load_checkpoint=args.ckpg)
    net.train(args.epochs)
    net.test()

```

#### 4.优化代码（2023.3.10更新）

代码原本更新参数的方式是一个样本一个样本更新。但这样训练太慢了，太丑陋了。于是我重写了代码，利用numpy可以进行一组样本一组样本的更新，极大提高了训练速度。

```python
	data,labels = self.split_data(self.train_data)
	data = np.c_[data, np.ones(data.shape[0])]
	self.forward(data,labels,eval=False)
```

并且代码也做了一定修改，实验报告就不放了。组更新结果如下，和单个样本更新结果几乎一样。

![image-20230310204908981](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimgimage-20230310204908981.png)

#### 5.可视化

使用`matplotlib`工具进行训练损失可视化。可见，模型在100epoch时候基本已经收敛。

![image-20230310225730946](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimgimage-20230310225730946.png)

## 四、心得体会

由于本次实验前做过CS231n的作业，除了一些框架构建外，其他做起来感觉还行，基本没啥大问题。

遇到的一个小问题是我在打乱的时候忘记把标签协同打乱了，导致最后结果很不好，也是一个粗心错误。

这里其实我为了增加拟合度，每次训练的epoch训练数据都是打乱的，但是最后验证效果居然反而没有不打乱好。或许是参数没有调整好或者测试数据太弱了吧。

无论如何，这一次实验又让我对机器学习底层细节进行了一定的复习，纯线性回归的难度并不算大，其实重难点还是公式的推导吧（想到CS231n的反向传播公式我就hh）。代码写的不算很优雅，想到啥写啥，后面有时间也会优化下代码的。