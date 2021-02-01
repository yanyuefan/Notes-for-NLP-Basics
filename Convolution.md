# Convolution

- 通过两个函数f,g生成第三个函数的数学算子
  - 表征函数f和
  - 经过翻转，平移的g
  - 的乘积函数围成的曲边梯形的面积
- 理解动态弹簧长度变换

## 序列角度

- 卷积是两个变量在某范围内相乘后求和的结果
- ![img](Convolution.assets/2f738bd4b31c8701160cdc36267f9e2f0608ffac.png)
- ![image-20210131182154315](Convolution.assets/image-20210131182154315.png)

## 函数角度公式

- <img src="Convolution.assets/image-20210131172155053.png" alt="image-20210131172155053" style="zoom:50%;" />
- 以上积分定义了一个新函数$h(x)=(f*g)(x)$
- ![img](Convolution.assets/v2-de38ad49f9a1c99dafcc5d0a7fcac2ef_720w.jpg)![img](Convolution.assets/v2-847a8d7c444508862868fa27f2b4c129_720w.jpg)
- 令 $\tau =x$ $n-\tau = y$ ，那么 ![[公式]](https://www.zhihu.com/equation?tex=x%2By%3Dn) 就是下面这些直线：
- <img src="Convolution.assets/v2-8be52f6bada3f7a21cebfc210d2e7ea0_hd.gif" alt="img" style="zoom:50%;" />

## 图片卷积计算过程

- ![img](Convolution.assets/v2-5ee9a99988137a42d1067deab36c4e51_720w.png)
- f目标计算矩阵，g算子。注意下标加和为1
- ![image-20210131174840016](Convolution.assets/image-20210131174840016.png)
- ![img](Convolution.assets/v2-c658110eafe027eded16864fb6a28f46_hd.gif)
- <img src="Convolution.assets/image-20210131175016457.png" alt="image-20210131175016457" style="zoom:60%;" />

## 卷积性质

- ![image-20210131223601395](Convolution.assets/image-20210131223601395.png)

## 和傅里叶变换的关系（爷今天不想看了Y）

- https://zhuanlan.zhihu.com/p/60638534
- 两函数的傅里叶变换的乘积等于它们卷积后的傅里叶变换
- 时域中的卷积对应于频域中的乘积
- 对于一个信号，时域越长，频域越短（集中）；时域越短，频域越长。
  - ![img](Convolution.assets/v2-b6308a5acd21dfb0c1fb2fc13d144e90_b.webp)
- [中心极限定理证明](https://www.cnblogs.com/TaigaCon/p/5014957.html)

