#coding: utf-8

# 1. 导入和设置

import torch
from torch import optim
from torch import nn, optim
from model import TextRNN
from cnews_loader import read_vocab, read_category, process_file
import numpy as np

train_file = 'cnews.train.txt'
test_file = 'cnews.test.txt'
val_file = 'cnews.val.txt'
vocab_file = 'cnews.vocab.txt'

def train(Train_Epoch):
    model = TextRNN().cuda()
    # 定义损失函数
    Loss = nn.MultiLabelSoftMarginLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    # 训练
    for epoch in range(Train_Epoch):
        print('epoch=', epoch)
        # 分批训练
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x = x_batch.cuda()
            y = y_batch.cuda()
            out = model(x)
            loss = Loss(out, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            print('loss=', loss)
            optimizer.step()
            # 计算准确率
            accuracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).cpu().numpy())
            print('accuracy:', accuracy)
        # 对模型进行验证
        if (epoch+1) % 5 == 0:
            for step, (x_batch, y_batch) in enumerate(val_loader):
                x = x_batch.cuda()
                y = y_batch.cuda()
                out = model(x)
                accuracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).cpu().numpy())
                if accuracy > best_val_acc:
                    torch.save(model, "model.pkl")
                    best_val_acc = accuracy
                    print('model.pkl saved')
                    print('val accuracy:', accuracy)
    return model


categories, cat_to_id = read_category() # 获取文本的类别及其对应id的字典
print(categories)

words, word_to_id = read_vocab('cnews.vocab.txt') # 获取训练文本中所有出现过的字及其所对应的id
print(words)

# 获取训练数据每个字的id和对应标签的one-hot形式
x_train, y_train = process_file(train_file, word_to_id, cat_to_id, 600)
print('x_train=', x_train)
x_val, y_val = process_file(val_file, word_to_id, cat_to_id, 600)

# GPU setting
import torch.utils.data as Data
cuda = torch.device('cuda')
x_train, y_train = torch.LongTensor(x_train), torch.Tensor(y_train)
x_val, y_val = torch.LongTensor(x_val), torch.Tensor(y_val)

# train param. setting

train_dataset = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(dataset = train_dataset, batch_size = 88, shuffle = True )

val_dataset = Data.TensorDataset(x_val, y_val)
train_loader = Data.DataLoader(dataset = train_dataset, batch_size = 88)
model = train()

