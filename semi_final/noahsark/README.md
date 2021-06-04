# Noah's Ark: ***N***EZHA Trained J***O***intly and Sep***A***rately with ***H***AN, L***S***TM and Deep Pyr***A***mid Convolutional Neu***R***al Networ***K***s

## 1 目录结构

- 深度学习部分：由学姐完成；
- NEZHA部分：由师兄提供baseline，本人进行优化并实现多种预训练、训练策略，并和师兄共同添加tricks及调参。

```

├── ./dl---------------------------------------深度学习部分
│   ├── ./dl/code
│   │   ├── ./dl/code/test
│   │   │   ├── ./dl/code/test/cvs_fusion.py------------------------------融合深度学习模型的结果
│   │   │   ├── ./dl/code/test/label1
│   │   │   │   ├── ./dl/code/test/label1/config.py-----------------------模型及训练参数
│   │   │   │   ├── ./dl/code/test/label1/directory.py--------------------数据集路径
│   │   │   │   ├── ./dl/code/test/label1/dpcnn_infer.py------------------DPCNN任务1推理
│   │   │   │   ├── ./dl/code/test/label1/dpcnn.py------------------------DPCNN模型定义
│   │   │   │   ├── ./dl/code/test/label1/han_infer.py--------------------HAN任务1推理
│   │   │   │   ├── ./dl/code/test/label1/han.py--------------------------HAN模型定义
│   │   │   │   ├── ./dl/code/test/label1/lstm_infer.py-------------------LSTM任务1推理
│   │   │   │   ├── ./dl/code/test/label1/lstm.py-------------------------LSTM模型定义
│   │   │   │   └── ./dl/code/test/label1/stopwords.txt-------------------停用词
│   │   │   └── ./dl/code/test/label2
│   │   │       ├── ./dl/code/test/label2/config.py-----------------------模型及训练参数
│   │   │       ├── ./dl/code/test/label2/directory.py--------------------数据集路径
│   │   │       ├── ./dl/code/test/label2/dpcnn_infer.py------------------DPCNN任务2推理
│   │   │       ├── ./dl/code/test/label2/dpcnn.py------------------------DPCNN模型定义
│   │   │       ├── ./dl/code/test/label2/han_infer.py--------------------HAN任务2推理
│   │   │       ├── ./dl/code/test/label2/han.py--------------------------HAN模型定义
│   │   │       ├── ./dl/code/test/label2/lstm_infer.py-------------------LSTM任务2推理
│   │   │       ├── ./dl/code/test/label2/lstm.py-------------------------LSTM模型定义
│   │   │       └── ./dl/code/test/label2/stopwords.txt-------------------停用词
│   │   └── ./dl/code/train
│   │       ├── ./dl/code/train/label1
│   │       │   ├── ./dl/code/train/label1/concat.py----------------------拼接初赛和复赛训练集，提取任务1标签
│   │       │   ├── ./dl/code/train/label1/config.py----------------------模型及训练参数
│   │       │   ├── ./dl/code/train/label1/directory.py-------------------数据集路径
│   │       │   ├── ./dl/code/train/label1/dpcnn_datasets.py--------------数据预处理
│   │       │   ├── ./dl/code/train/label1/dpcnn.py-----------------------DPCNN模型定义
│   │       │   ├── ./dl/code/train/label1/dpcnn_train.py-----------------DPCNN任务1训练
│   │       │   ├── ./dl/code/train/label1/han_datasets.py----------------数据预处理
│   │       │   ├── ./dl/code/train/label1/han.py-------------------------HAN模型定义
│   │       │   ├── ./dl/code/train/label1/han_train.py-------------------HAN任务1训练
│   │       │   ├── ./dl/code/train/label1/lstm_datasets.py---------------数据预处理
│   │       │   ├── ./dl/code/train/label1/lstm.py------------------------LSTM模型定义
│   │       │   ├── ./dl/code/train/label1/lstm_train.py------------------LSTM任务1训练
│   │       │   ├── ./dl/code/train/label1/seed.py------------------------设定随机种子
│   │       │   └── ./dl/code/train/label1/stopwords.txt------------------停用词
│   │       └── ./dl/code/train/label2
│   │           ├── ./dl/code/train/label2/config.py----------------------模型及训练参数
│   │           ├── ./dl/code/train/label2/directory.py-------------------数据集路径
│   │           ├── ./dl/code/train/label2/dpcnn_datasets.py--------------数据预处理
│   │           ├── ./dl/code/train/label2/dpcnn.py-----------------------DPCNN模型定义
│   │           ├── ./dl/code/train/label2/dpcnn_train.py-----------------DPCNN任务2训练
│   │           ├── ./dl/code/train/label2/han_datasets.py----------------数据预处理
│   │           ├── ./dl/code/train/label2/han.py-------------------------HAN模型定义
│   │           ├── ./dl/code/train/label2/han_train.py-------------------HAN任务2训练
│   │           ├── ./dl/code/train/label2/lstm_datasets.py---------------数据预处理
│   │           ├── ./dl/code/train/label2/lstm.py------------------------LSTM模型定义
│   │           ├── ./dl/code/train/label2/lstm_train.py------------------LSTM任务2训练
│   │           ├── ./dl/code/train/label2/seed.py------------------------设定随机种子
│   │           └── ./dl/code/train/label2/stopwords.txt------------------停用词
│   ├── ./dl/prediction_result
│   │   ├── ./dl/prediction_result/dpcnnMerge2label.py--------------------拼接DPCNN任务1和任务2的预测结果
│   │   ├── ./dl/prediction_result/hanMerge2label.py----------------------拼接HAN任务1和任务2的预测结果
│   │   ├── ./dl/prediction_result/label1
│   │   ├── ./dl/prediction_result/label2
│   │   └── ./dl/prediction_result/lstmMerge2label.py---------------------拼接LSTM任务1和任务2的预测结果
│   └── ./dl/user_data
│       └── ./dl/user_data/model_data
│           ├── ./dl/user_data/model_data/label1
│           └── ./dl/user_data/model_data/label2
├── ./Dockerfile-----------------------定制镜像
├── ./merge.py-------------------------融合NEZHA和深度学习的预测结果
├── ./nezha----------------------------NEZHA预训练+微调部分
│   ├── ./nezha/corpus_vocab.py-------------------------------------------制作语料库和词典
│   ├── ./nezha/create_pretraining_data.py--------------------------------构建预训练格式数据
│   ├── ./nezha/directory.py----------------------------------------------数据集、中间文件、生成文件等路径
│   ├── ./nezha/helper
│   │   ├── ./nezha/helper/adv_training.py--------------------------------Trick:对抗训练
│   │   ├── ./nezha/helper/data_generator.py------------------------------数据生成器
│   │   ├── ./nezha/helper/preprocess.py----------------------------------数据预处理
│   │   ├── ./nezha/helper/seed.py----------------------------------------设定随机种子
│   │   └── ./nezha/helper/warmup_cosine_decay.py-------------------------Trick:Warm Up余弦退火
│   ├── ./nezha/merge.py--------------------------------------------------融合分开训练与联合训练的结果
│   ├── ./nezha/nsp_corpus_vocab.py---------------------------------------制作NSP任务的语料库和词典
│   ├── ./nezha/pretrain--------------------------------------------------预训练参数及辅助代码
│   │   ├── ./nezha/pretrain/bert_config.json
│   │   ├── ./nezha/pretrain/gpu_environment.py
│   │   ├── ./nezha/pretrain/modeling.py
│   │   ├── ./nezha/pretrain/optimization.py
│   │   └── ./nezha/pretrain/tokenization.py
│   ├── ./nezha/pretraining.py--------------------------------------------预训练
│   ├── ./nezha/run.py----------------------------------------------------联合训练融合分开训练运行脚本
│   ├── ./nezha/separate_category_predict.py------------------------------分开训练策略:任务2推理
│   ├── ./nezha/separate_category_train_val.py----------------------------分开训练策略:任务2微调训练
│   ├── ./nezha/separate_predict.py---------------------------------------拼接任务1和任务2的预测结果
│   ├── ./nezha/separate_region_predict.py--------------------------------分开训练策略:任务1推理
│   ├── ./nezha/separate_region_train_val.py------------------------------分开训练策略:任务1微调训练
│   ├── ./nezha/joint_predict.py------------------------------------------联合训练策略:推理
│   └── ./nezha/joint_train_val.py----------------------------------------联合训练策略:微调训练
└── ./run.sh---------------------------完整方案全流程运行脚本
```

## 2 线上环境

- 镜像地址：registry.cn-shanghai.aliyuncs.com/tcc-public/tensorflow:1.13.1-cuda10.0-py3

- 操作系统：Ubuntu 18.04

- 显卡及显存：NVIDIA V100 (16GB)

- CUDA版本：10.0

- Python版本：3.5.2

- Python依赖：

  ```
  bert4keras==0.10.0
  keras==2.3.1
  tensorflow-gpu==1.15.0
  scikit-learn==0.22.2
  torch==1.4.0
  h5py==2.10.0
  pandas==0.24.2
  iterative-stratification==0.1.6
  tqdm==4.60.0
  ```

## 3 实验结果

### 3.1 A榜

#### 3.1.1 NEZHA & BERT

| 模型 | 训练策略 | 余弦退火 | seq长度 | 预训练步数 | 训练fold | 训练epoch | 任务1得分 | 任务2得分 | 总得分 |
| ---- | ---------- | ---------- | ---------- | ---------- | ------- | ------- | ------- | ------- | ---- |
| NEZHA | 分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 8 | 联合:10<br/>分开:5 | 0.9398 | 0.9388 | 0.9394 |
| NEZHA:0.7<br>BERT:0.3 | 分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 5 | 联合:10<br>分开:5 | 0.9399 | 0.9369 | 0.9387 |
| NEZHA | 分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 5 | 联合:10<br/>分开:5 | 0.9395 | 0.9367 | 0.9384 |
| NEZHA | 分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 5 | 联合:10<br/>分开:5 | 0.9401 | 0.9347 | 0.9379 |
| NEZHA | 分开训练 | 是 | 100 | 30000 | 5 | 5 | 0.9369 | 0.9349 | 0.9361 |
| NEZHA | 预训练加入分组NSP任务<br>分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 5 | 联合:10<br/>分开:5 | 0.9363 | 0.9318 | 0.9345 |
| BERT | 分开:0.7<br/>联合:0.3 | 是 | 100 | 30000 | 5 | 联合:10<br/>分开:5 | 0.9350 | 0.9288 | 0.9325 |
| NEZHA | 分开训练 | 否 | 100 | 30000 | 5 | 5 | 0.9326 | 0.9302 | 0.9316 |
| NEZHA | 分开训练 | 否 | 100 | 30000 | 10 | 10 | 0.9314 | 0.9264 | 0.9294 |
| NEZHA | 联合训练 | 否 | 100     | 100000 | 5   | 5  | 0.9257 | 0.9142 | 0.9212 |
| NEZHA | 联合训练 | 否 | 100 | 30000 | 5 | 5 | 0.9266 | 0.9099 | 0.9200 |

#### 3.1.2 DL

| 模型                               | 训练策略 | seq长度 | 训练fold | 训练epoch | 任务1得分 | 任务2得分 | 总得分 |
| ---------------------------------- | -------- | ------- | -------- | --------- | --------- | --------- | ------ |
| HAN:0.5<br/>DPCNN:0.3<br/>LSTM:0.2 | 分开训练 | 55      | 10       | 15        | -         | -         | 0.9306 |
| HAN                                | 分开训练 | 55      | 10       | 15        | -         | -         | 0.9231 |
| DPCNN                              | 分开训练 | 55      | 10       | 15        | -         | -         | 0.9202 |
| LSTM                               | 分开训练 | 55      | 10       | 15        | 0.9217    | 0.9160    | 0.9195 |
| LSTM                               | 分开训练 | 100     | 15       | 30        | 0.9205    | 0.9178    | 0.9194 |
| DPCNN                              | 联合训练 | 55      | 15       | 30        | 0.9255    | 0.9049    | 0.9173 |
| LSTM                               | 分开训练 | 70      | 5        | 10        | 0.9171    | 0.9030    | 0.9115 |
| LSTM                               | 联合训练 | 55      | 15       | 30        | 0.9181    | 0.9006    | 0.9111 |
| LSTM                               | 分开训练 | 55      | 8        | 15        | 0.9153    | 0.9019    | 0.9099 |

#### 3.1.3 融合

| 模型                 | NEZHA训练策略         | DL训练策略 | 余弦退火 | seq长度             | 预训练步数 | 训练fold | 训练epoch          | 任务1得分 | 任务2得分 | 总得分 |
| -------------------- | --------------------- | ---------- | -------- | ------------------- | ---------- | -------- | ------------------ | --------- | --------- | ------ |
| NEZHA:0.8<br/>DL:0.2 | 分开:0.7<br/>联合:0.3 | 分开训练   | 是       | NEZHA:100<br/>DL:55 | 30000      | 5        | 联合:10<br/>分开:5 | 0.9408    | 0.9383    | 0.9398 |
| NEZHA:0.7<br/>DL:0.3 | 分开:0.7<br/>联合:0.3 | 分开训练   | 是       | NEZHA:100<br/>DL:55 | 30000      | 5        | 联合:10<br/>分开:5 | 0.9397    | 0.9381    | 0.9391 |

### 3.2 B榜

| 模型                 | NEZHA训练策略         | DL训练策略 | 余弦退火 | seq长度             | 预训练步数 | NEZHA训练fold | NEZHA训练epoch     | DL训练fold | DL训练epoch | 任务1得分 | 任务2得分 | 总得分 |
| -------------------- | --------------------- | ---------- | -------- | ------------------- | ---------- | ------------- | ------------------ | ---------- | ----------- | --------- | --------- | ------ |
| NEZHA:0.8<br/>DL:0.2 | 分开:0.8<br/>联合:0.2 | 分开训练   | 是       | NEZHA:100<br>DL:55  | 30000      | 8             | 联合:10<br/>分开:5 | 10         | 15          | 0.9433    | 0.9412    | 0.9425 |
| NEZHA:0.8<br/>DL:0.2 | 分开:0.8<br/>联合:0.2 | 分开训练   | 是       | NEZHA:100<br/>DL:55 | 50000      | 8             | 联合:10<br/>分开:5 | 15         | 15          | 0.9433    | 0.9408    | 0.9423 |
