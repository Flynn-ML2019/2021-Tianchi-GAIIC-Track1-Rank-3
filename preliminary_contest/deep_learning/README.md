# 深度学习模型

## 1 执行环境

操作系统：Ubuntu 20.04.1

cuda版本：11.2

Python版本：3.8

Python依赖：执行`pip3 install -r requirements.txt`命令安装。

## 2 实验结果

| 模型 | 数据增强 | 词向量种类 | 词向量轮数 | 词向量维度 | 词向量冻结 | seq长度 | fold | 线上得分 |
| ------- | ---------- | ---------- | ---------- | ---------- | ------- | ---- | ---------- | ---------- |
| RCNN + RCNNAttn + DPCNN | 否  | concat     | 50         | 100        | 是         | 100     | 20   | 0.9036     |
| RCNN   | 否  | concat     | 50         | 100        | 是         | 100     | 20   | 0.9011     |
| RCNN  | 否 | 三种融合     | 50         | 100        | 是         | 100     | 20   | 0.9008   |
| RCNN + RCNNAttn + DPCNN | 否 | concat | 50 | 100 | 是 | 100 | 5 |  0.8999|
| RCNN   | 否  | word2vec   | 50         | 100        | 是         | 100     | 20   | 0.8996     |
| RCNN   | 否  | concat     | 50         | 200        | 是         | 100     | 20   | 0.8993     |
| RCNN   | 否  | concat     | 50         | 100        | 是         | 50      | 20   | 0.8977     |
| RCNNAttn | 否 | concat | 50 | 100 | 是 | 100 | 20 | 0.8966 |
| RCNN   | 否  | concat     | 100        | 100        | 是         | 100     | 25   | 0.8960     |
| RCNN   | 否  | glove      | 50         | 100        | 否         | 50      | 10   | 0.8946     |
| RCNN   | 否  | concat     | 50         | 100        | 是         | 50      | 10   | 0.8943     |
| RCNN  | 是 | concat     | 50         | 100        | 是         | 100     | 10   |   0.8918   |
| DPCNN   | 否  | concat     | 50         | 100        | 是         | 100     | 20   | 0.8895     |
