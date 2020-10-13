# Attention and Augmented Recurrent Neural Networks

- https://distill.pub/2016/augmented-rnns/

## Neural Turing Machine

- https://arxiv.org/abs/1410.5401
- 将RNN和一个外部memory bank结合
- 组成
  - nn控制器(controller)
  - 模拟器或处理器(读取磁带设备)
  - 外部存储memory

![img](https://image.jiqizhixin.com/uploads/editor/a1eab88f-6908-4f6e-b5ce-6de12fdd9e70/1530498428259.png)

- ![用处](https://image.jiqizhixin.com/uploads/editor/626b4205-1d9e-484d-ba5a-11f4969ae3f1/1530498428710.png)
  - bAbI数据集 推理能力
- 问题
  - Architecture dependent
    - 对于每一个给定的输入或输出单元，在给定的时间步长下你能够读取或者写入多少向量
  - 大规模参数
  - 不能从 GPU 加速中受益
    - 当下所做的都是基于之前的输入。很难将这些部分分解成容易的并行计算
  - 难以训练
    - 数值不稳定性
      - 学习算法
      - 通用方法
        - Gradient clipping
          - 限制参数训练速度
          - 对学习long range dependency有帮助
        - Loss clipping
          - 给能够改变的参数总和设置一个上限
          - 神经图灵机远离它们的目标
    - 很难使用记忆memory
    - 需要很好地优化
    - 实际中很难使用

- 拓展
  - 可微分神经计算机
    - 放弃了基于索引移动(沿着记忆或者磁带的移动)的寻址方式
    - 尝试基于它们看到的东西直接在记忆中搜索给定的向量
      - 有分配记忆和释放记忆的能力