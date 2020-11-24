## 发展历史

- 早期方法
  - 基于规则
  - 基于字典
- 传统机器学习
  - HMM
  - MEMM
  - CRF
- 深度学习
  - RNN-CRF
  - CNN-CRF
- 近期方法
  - 注意力模型
  - 迁移学习
  - 半监督学习

## 基础知识

### 数据标注

- BIO

- BIOES

  - B，即Begin，表示开始

    I，即Intermediate，表示中间

    E，即End，表示结尾

    S，即Single，表示单个字符

    O，即Other，表示其他，用于标记无关字符

### 方法

#### HMM

- 隐马尔科夫模型(Hidden Markov Model 生成模型 概率图模型)
  - 学习任务：计算变量的概率分布
  - 问题描述组成
    - **观测序列**+不能观察到的隐藏状态序列，简称**状态序列**
    - ![img](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONykTKBgL1LqZbYkegrsiaA5PACnonz2rzboqiaE8hRF3rhqmicicYAqvpcmQPtmfbewmJwRtqMeWfV5ia7Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
      - 通过考虑联合概率分布P(Y,X)推断Y的分布
  - 定义
    - 对于一个长度为T的序列，**I对应的状态序列**, **O是对应的观察序列**，即: $I={i_1,i_2,...,i_T},O={o_1,o_2,...o_T}$ 其中，任意一个隐藏状态$i_t∈Q$,任意一个观察状态$o_t∈V$
    - HMM假设（定义）
      - 齐次马尔科夫链假设
        - 任意时刻的隐藏状态只依赖于它前一个隐藏状态
        - 描述
          - 时刻 $t$ 的隐藏状态是 $i_t=q_i$, 在时刻 $t+1$ 的隐藏状态是$i_{t+1}=q_j$ , 
          - 则时刻$t$到时刻$t+1$的HMM状态转移概率$a_{ij}$：$a_{ij}=P(i_{t+1}=q_j|i_t=q_i)$
          -   $a_{ij}$可以组成马尔科夫链的**隐藏状态状态转移矩阵A**: A=$[a_{ij}]_{N*N}$
      - 时刻t=1, **隐藏状态概率分布矩阵Π**：$Π=[\pi(i)]_N$ $\pi(i)=P(i_1 = q_i)$
      - 观测独立性假设
        - 任意时刻的**观察状态O**只仅仅依赖于当前时刻的**隐藏状态I**
        - 观测状态生成概率矩阵**B**： $b_j(k)=P(o_t=v_k|i_t=q_j)$
    - 一个HMM模型，由<u>隐藏状态初始概率分布Π</u>, <u>状态转移概率矩阵A</u>和<u>观测状态概率矩阵B</u>决定。A, Π决定状态序列，B决定观测序列。由一个三元组λ表示：$λ=(A,B,Π)$
  - 需解决问题
    - 即给定模型$$λ=(A,B,Π)$$和观测序列$$O={o1,o2,...oT}$$，计算在模型λ下**观测序列O**出现的概率P(O|λ), 前向后向算法
    - 给定观测序列$$O={o1,o2,...oT}$$，估计模型$$λ=(A,B,Π)$$的参数，使得该模型下观测序列的条件概率P(O|λ)最大。 EM算法
    - 给定模型$$λ=(A,B,Π)$$和观测序列$$O={o1,o2,...oT}$$，求给定观测序列条件下，最可能出现的对应的状态序列

- [马尔科夫随机过程](https://blog.csdn.net/DeepOscar/article/details/81036635?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)
  
  - X在 **$t_n$** 时刻的状态只与其前一时刻时状态的值有关
    - ​	图n的边用从时刻n的状态到时刻n+1的状态的概率$Pr(X_{n+1} = x |X_n=x_n)$来标记
  - 状态的改变叫转移，与不同的状态改变相关的概率叫做转移概率
  - 片段
  - 细致平稳条件(Detailed Balance Condition)
    - 马尔科夫链、分布$\pi$ 和概率转移矩阵 P
    - $\pi_i P_{ij} = \pi _j P_{ji}$
  
- HMM在NLP（分词上的应用）

  - 随机变量Y在t时刻的状态仅由y(t-1)决定，观测序列变量X在t时刻的状态仅由yt决定，有：

    <img src="https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONykTKBgL1LqZbYkegrsiaA5PABJt2Anr5WU4AER4SlUyHqfbDhQtnEjRRzYB3bibv6RD2So4bI0QkIEw/640?wx_fmt=png&amp;tp=webp&amp;wxfrom=5&amp;wx_lazy=1&amp;wx_co=1" alt="img" style="zoom:60%;" />

    从而可以推出联合概率：

    ![img](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONykTKBgL1LqZbYkegrsiaA5PA5MVoajMADAicIZfCAJJ5XqPbmWadlzSxlbD8vMtdSYtyibKbfFtDxzPw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

    - $=P(Π)P(B_{1,1})\prod_{i=2}^n A_{i,i-1}B_{i,i}$
      - P(yi=”E”|yi-1=”M”)
      - P(Xi=”今”|yi=”B”)描述的是i时刻标记为“B”时，i时刻观测到到字为“深”的概率

  - 例子

    - 请问今天的天气怎么样 观测序列
    - BEBESBEBIE 状态序列

  - 处理过程

    - 计算状态序列算法
      - ![img](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONykTKBgL1LqZbYkegrsiaA5PAfNcL565WoOPa76Q8Wrtp4npMKDoRiafB40uP9ldoceK6W6Ib6biatMJA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
      - ![img](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONykTKBgL1LqZbYkegrsiaA5PAVxsBeBeo0ZiabxUl28j22snUibBrVUG6FllooyiaoaGJRvJWGJECWMZicQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
      - 当语料确定时，![img](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONykTKBgL1LqZbYkegrsiaA5PA62XBBZjjDiboNHibbbfJlsXObG9yGgQ7zKLcIYmqSLtZdyNDnwLctabg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)只需计算
      - ![img](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONykTKBgL1LqZbYkegrsiaA5PA5MVoajMADAicIZfCAJJ5XqPbmWadlzSxlbD8vMtdSYtyibKbfFtDxzPw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
    - 维特比算法
      - 

#### CRF

- 条件随机场

## 模型

## 工具

### Stanford NER

- 基于条件随机场的命名实体识别系统
- [官网](https://nlp.stanford.edu/software/CRF-NER.shtml) | [GitHub 地址](https://github.com/Lynten/stanford-corenlp)

### MALLET

- 统计自然语言处理的开源包，其序列标注工具的应用中能够实现命名实体识别
- [官网](http://mallet.cs.umass.edu/)

### Hanlp

- 一系列模型与算法组成的NLP工具包，由大快搜索主导并完全开源
- [官网](http://hanlp.linrunsoft.com/) | [GitHub 地址](https://github.com/hankcs/pyhanlp)

### NLTK

- [官网](http://www.nltk.org/) | [GitHub 地址](https://github.com/nltk/nltk)

### SpaCy

- [官网](https://spacy.io/) | [GitHub 地址](https://github.com/explosion/spaCy)



### Crfsuite

- 载入自己的数据集去训练CRF实体识别模型

- [文档](https://sklearn-crfsuite.readthedocs.io/en/latest/?badge=latest ) | [GitHub 地址](https://github.com/yuquanle/StudyForNLP/blob/master/NLPbasic/NER.ipynb)

  

