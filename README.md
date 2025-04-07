# SMS Spam Classifier

这是一个基于机器学习的短信垃圾分类系统，可以自动识别短信是否为垃圾信息。该系统使用TF-IDF特征和神经网络模型进行分类，并提供REST API服务。

## 项目结构

```
sms_spam_classifier/
├── data_processor.py  # 数据处理模块
├── train.py          # 模型训练模块
├── api.py           # API服务模块
├── requirements.txt  # 项目依赖
└── SMSSpamCollection # 训练数据集
```

## 功能特点

- 文本预处理：移除URL、特殊字符，转换为小写等
- TF-IDF特征提取：使用scikit-learn的TfidfVectorizer
- 神经网络分类器：使用PyTorch实现的简单前馈神经网络
- REST API服务：使用FastAPI实现的Web服务
- 模型评估：提供准确率、精确率、召回率等多个评估指标

## 性能指标

在测试集上的表现：
- 准确率：98.12%
- 垃圾短信（Spam）：
  - 精确率：96%
  - 召回率：89%
  - F1分数：93%
- 正常短信（Ham）：
  - 精确率：98%
  - 召回率：99%
  - F1分数：99%

## 安装说明

1. 克隆项目并安装依赖：
```bash
pip install -r requirements.txt
```

2. 训练模型：
```bash
python train.py
```

3. 启动API服务：
```bash
python api.py
```

## API使用说明

### 健康检查
```bash
GET /health
```
返回服务状态。

### 短信分类
```bash
POST /classify
Content-Type: application/json

{
    "text": "您的短信内容"
}
```

返回示例：
```json
{
    "text": "原始文本",
    "is_spam": true,
    "confidence": 0.95,
    "preprocessed_text": "预处理后的文本"
}
```

## 技术栈

- Python 3.13
- PyTorch：深度学习框架
- scikit-learn：特征工程
- FastAPI：Web框架
- pandas：数据处理
- numpy：数值计算

## 开发说明

### 数据处理流程

1. 文本预处理（data_processor.py）：
   - 转换为小写
   - 移除URL
   - 移除特殊字符
   - 移除多余空格

2. 特征提取：
   - 使用TF-IDF向量化
   - 固定特征维度为768
   - 自动填充或截断到指定长度

### 模型架构

简单的前馈神经网络：
- 输入层：768维（TF-IDF特征）
- 隐藏层：256维，ReLU激活
- Dropout层：防止过拟合
- 输出层：2维（正常/垃圾）

### 训练参数

- 批次大小：16
- 学习率：0.001
- 训练轮数：3
- 优化器：Adam
- 损失函数：CrossEntropyLoss

## 注意事项

1. 首次运行时需要下载并处理数据集
2. 模型文件（best_model.pt）会在训练后自动保存
3. API服务默认在8000端口运行
4. 建议在生产环境中使用gunicorn或其他WSGI服务器 