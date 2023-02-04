# mini-ssd

## ssd 损失函数
下面来介绍如何设计损失函数。

将总体的目标损失函数定义为 定位损失（loc）和置信度损失（conf）的加权和：

$$
L(x,c,l,g) = \frac{1}{N}(L_{conf}(x,c)+\alpha L_{loc} (x,l,g)) (1)
$$

其中N是匹配到GT（Ground Truth）的prior bbox数量，如果N=0，则将损失设为0；而 **α** 参数用于调整confidence loss和location loss之间的比例，默认 **α=1**。

confidence loss是在多类别置信度(c)上的softmax loss，公式如下：

$$
L_{conf}(x,c) = -\sum_{i \in Pos}^N x^{p}_{ij} log(\hat{c}^{p}_{i}) - \sum_{i \in Neg} log(\hat{c}^{0}_{i})  
$$
$$
Where \hat{c}^{p}_{i} = \frac{exp(c^{p}_{i})}{\sum_p exp(c^{p}_{i})} (2)
$$

其中i指代搜索框序号，j指代真实框序号，p指代类别序号，p=0表示背景。其中$x^{p}_{ij}=\left\{1,0\right\}$ 中取1表示第i个prior bbox匹配到第 j 个GT box，而这个GT box的类别为 p 。$C^{p}_{i}$ 表示第i个搜索框对应类别p的预测概率。此处有一点需要关注，公式前半部分是正样本（Pos）的损失，即分类为某个类别的损失（不包括背景），后半部分是负样本（Neg）的损失，也就是类别为背景的损失。

而location loss（位置回归）是典型的smooth L1 loss

$$
L_{loc}(x,l,g) = \sum_{i \in Pos  m \in \left\{c_x,c_y,w,h\right\}}^N \sum x^{k}_{ij} smooth_{L1}(l^{m}_{i}-\hat{g}^{m}_{j}) (3)
$$

$$
\hat{g}^{c_x}_{j}=(g^{c_x}_{j}-d^{c_x}_{i})/d^{w}_{i}
$$

$$
\hat{g}^{c_y}_{j}=(g^{c_y}_{j}-d^{c_y}_{i})/d^{h}_{i}
$$

$$
\hat{g}^{w}_{j}=log(\frac{g^{w}_{j}}{d^{w}_{i}})
$$

$$
\hat{g}^{h}_{j}=log(\frac{g^{h}_{j}}{d^{h}_{i}})
$$
注意：位置损失当且仅当一个prior bbox  与gt bbox 预测的类别为一样时才进行计算    

其中，l为预测框，g为ground truth。(cx,xy)为补偿(regress to offsets)后的默认框d的中心,(w,h)为默认框的宽和高。更详细的解释看-看下图：

<div align=center>
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter03/3-33.jpg">
</div>

# My mini SSD
## 模型结构
### 预测特征层
预测特征层只有一个：4x3     
在4x3的先验格子中，一个格子不会出现2个目标的情况    
正样本：4x3格子中有目标的格子   
负样本：4x3格子中无目标的格子   
## 分类头和回归头
### 边界框编码
$$
    d_x = \frac{c_x-x_0}{w} 
$$
$$
    d_y = \frac{c_y-y_0}{h}
$$
$d_x,d_y为目标中心相对于先验格子左上角偏移量的归一化值。c_x,c_y为目标的中心坐标。$
$x_0,y_0为先验格子的左上角的坐标。w，h为先验格子的宽和高。$
### 分类头和回归头结构
+ 分类头：因为只有目标和背景两类，使用sigmoid激活所以分类头的输出为 (batch size,3,4,1)。
+ 回归头：因为只预测$c_x,c_y$两个值，所以回归头输出为 (batch size,3,4,2)。


## 损失函数  
将总体的目标损失函数定义为 定位损失（loc）和置信度损失（conf）的加权和：
$$
\begin{equation}
L(x,c,l,g) = \frac{1}{N}(L_{conf}(x,c)+\alpha L_{loc} (x,l,g))
\end{equation}
$$
$其中N为真实值中有目标的格子数，如果N=0，则将损失设为0；$
$而 α 参数用于调整置信度损失和定位损失之间的比例，默认 α=1。$
### 置信度损失  
因为为sigmoid激活后的二分类任务，使用交叉熵损失函数    
$$
\begin{equation}
    % L_{conf}(x,c) = -\sum_{i \in Pos}^N  log(\hat{c}_{i}) - \sum_{i \in Neg}^M log(\hat{c}_{i}) 
    L_{conf}(x,c) = -\sum (\ ylog(\hat{c_i}) +(1-y)log(\hat{c_i}) \ )
\end{equation}
$$
$$
    \hat{c_i} = Sigmoid(c_i)
$$
<!-- $Pos定义为格子上有目标的样本，Neg定义为格子上没有目标的样本。这里的有无目标指的是真实值，而非预测值。$ -->
$y=\left\{1，0\right\},当标签上有目标时为1，无目标时为零。其实就是Binary \ Crossentropy 损失。$

### 位置损失  
当且仅当预测与真实图片在某个格子内有目标时才进行计算（回归结果不在格子内也算，这就体现出Smooth L1 的价值了）。  
偏差为与格子中心进行归一化后的偏差（除以格子的边长）      
使用Smooth L1 loss  
生成的标签为目标中心相对于格子左上角的偏差，需要在损失函数内进行变换    
$$
\begin{equation}
    L_{loc}(x,l,g) = \sum_{i \in Pos }^N \sum_{m \in \left\{c_x,c_y\right\}} x_i smooth_{L1}(l_i^{m}- \hat{g}_i^{m})
\end{equation}
$$
$其中l为预测值的回归偏移量归一化值，\hat{g}为真实值的回归偏移量归一化值。$  
$x_i=\left\{1，0\right\}，当且仅当标签与预测都有目标时为1。$
### 正负样本比例
ssd中利用Hard negative mining方法得到1:3的正负样本比例。这里为了方便，直接利用数据生成器生产1:3正负样本比例的数据集，暂时不在损失函数部分实现Hard negative mining。

