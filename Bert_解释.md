# What does BERT learn about the structure of language

## 摘要

- BERT’s phrasal representation captures
  **phrase-level information** in the **lower layers**.
- BERT’s **intermediate layers** encode a rich hierarchy of **linguistic information**
  - surface features at the bottom
  - syntactic features in the middle 
  - semantic features at the top
- BERT  require deeper layers when long-distance dependency information is required
  - track subject-verb agreement
- BERT representations capture linguistic information in a compositional way that mimics classical, tree-like structures
  - 以模仿经典的**树状结构**的合成方式捕获语言信息

## 介绍

- (Goldberg. 2019)BERT captures **syntactic phenomena** well when evaluated on its ability to **track subject-verb agreement**.
  - arXiv:1901.05287. Version 1.
- 实验
  - 低层捕捉短语级信息，这些信息在高层中会被稀释
  - 使用探针任务表明BERT捕获了多层的语言信息(Conneau et al. (2018), What you can cram into a single n$&!#* vector: Probing sentence embeddings for linguistic properties.)
  - 通过测试BERT表示在追踪主语和动词的一致性任务的能力，发现BERT需要更多层来执行包含长距离依赖的困难任务
  - 使用TPDN探索关于BERT表示的不同假设并发现BERT captures classical, tree-like structures
    - Tensor Product Decomposition Network ((McCoy et al., 2019)RNNs Implicitly Implement Tensor Product Representations. ICLR)

## 短语语法

- 给定token sequence $s_i,...,s_j$，计算每层 $l$ 的span 表示$s_{(s_i,s_j), l}$ 
  - 将隐层向量($h_{s_i,l}$)、($h_{s_j, l}$)、它们的element-wise product、它们的差 **concatenate**
- 在CoNLL 2000 chunking dataset随机选取了3000标注后的chunks和500个未标注的spans
- 可视化方法
  - span 表示
  - t-SNE
    - 可视化高维数据的一种非线性dimensionality reduction方法
  - NMI矩阵
- 结论
  - 底层捕获了phrase-level信息，这些信息在高层被稀释

##探针任务

- 使用模型的输出作为辅助任务的输入，如果辅助分类任务可以很好的预测语言属性(linguistic property)，说明原始模型可以很好的编码该属性。
- 分类
  - surface tasks
    - sentence length（`SentLen`）
    - presence of words in the sentence (`WC`)
  - Syntactic tasks
    - sensitivity to word order (`BShift`)
    - the depth of the syntactic tree (`TreeDepth`)
    - the sequence of top-level constituents in syntax tree (`TopConst`)
  - Semantic tasks
    - tense (`Tense`)
    - subject number in main clause (`SubjNum`)
    - the sensitivity to random replacement of a noun/verb (`SOMO`)
    - the random swapping of coordinated clausal conjuncts (`CoordInv`) 并列子句的随机交换
- use the `SentEval toolkit` 寻找最好的探针分类器
- random encoder 可以编码大量的词汇和结构信息（1809.10040）
- 对照组
  - 未训练的Bert（参数设置为随机值）
- 结论：surface information at the bottom, syntactic information in the middle, semantic information at the top
- 反直觉
  - 在`SentLen`任务中未训练过Bert的高层比训练后版本表现好
  - 说明：untrained models contain sufficient information to predict a basic surface feature，训练后版本预测basic surface feature的能力会弱化

## 主语动词一致性检验(Subject-Verb Agreement)

- 目标：检验神经网络是否能编码语法结构信息
- extend Goldberg(2019)的工作
  - performing the test on each layer of
    BERT
  - controlling for the number of attractors
- 结论
  - the middle layers perform well in most cases
  - syntactic features were captured well in the middle layers
  - higher BERT layers is able to handle the long-distance dependency

## 组成结构(compositional Structure)

- Tensor Product Decomposition Networks (TPDN)
  - input token representations 
    - 用tensor product sum预先选定的role scheme
- 作者猜想
  - 对于给定的role scheme ,如果TPDN训练结果近似于神经网络学习的表示，则该role scheme 可能能表示模型学到的组成结构
- 实验
  - For each BERT layer, we work with five different role schemes. Each word’s role is computed based on 
    - its left-to-right index,
    - its right-to-left index, 
    - an ordered pair containing its left-to-right and right-to-left indices, 
    - its position in a syntactic tree with no unary nodes（一元节点） and no labels
    - an index common to all the words in the sentence (bag-of-words)忽略位置信息
  - define a role scheme based on random binary trees.
  - 数据集、任务
    - SNLI corpus、premise sentence
  - 过程
    - 用来自Bert的word embedding初始化TPDN的token embedding
    - 冻结
    - 学习他们之间的线性映射
    - 使用MSE loss function
  - 结果
    - 底层layers的role scheme是R2L/L2R
    - 中高层的是tree
- case study on dependency trees induced from self attention weight
  - fixing the gold root as the starting node
  - 使用Chu-Liu-Edmonds algorithm
    - maximum spanning tree algorithm
  - 从2层11个注意力头处获得示例句子中每个词对的自注意力权重，由此推断句中依存关系
  - 结果
    - 可以精确捕捉到
      - determiner-noun限定词-名词
      - subject-verb 主语-动词
    - 部分捕捉到了推断-论据结构

## 相关工作

- Peters et al. (2018)
  - 问题：预训练模型选择对下游任务精确度的影响；学到的上下文词表示的特性
  - 结论：学的都比标准词嵌入(`GloVe`)好，这些结构分层学习了语言学特征
- Goldberg(2019)
  - BERT model captures syntactic information well for subject-verb agreement.
  - Extend 结论
    - BERT requires deeper layers for handling harder cases involving long-distance dependency information.
- Tenney et al. (2019)
  - a novel edge probing task
    - 上下文词表示法如何在一系列句法，语义，局部和长程现象中编码句子结构。
  - 结论
    - 经过语言建模和机器翻译训练的上下文单词表示法可以很好地编码句法现象，但在语义任务上相较于非上下文基准给出的改进有限
- Liu et al. (2019)
  - 上下文词嵌入没有学到细粒度的语言学知识
  - 高层的RNN会面向任务，而Transformer没这个现象
  - 在相似任务上进行预训练的性能比语言建模预训练要好
- Hewitt and Manning (2019)
  - 可以从上下文词表示的线性转换中恢复语法树，比非上下文基准效果好
  - BERT的组合模型模仿了传统的句法分析。

# What Does BERT Look At?
An Analysis of BERT’s Attention