#coding: utf-8

# Part 1: 导入
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tnrange
import numpy as np
from tensorflow import keras as kr
np.random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.backends.cudnn.determindistic=True
torch.backends.cudnn.benchmark = False

print('ready')

train_file = 'cnews.train.txt'
test_file = 'cnews.test.txt'
val_file = 'cnews.val.txt'
vocab_file = 'cnews.vocab.txt'

# 准备数据 （数值向量由文本转入）
# 读取词汇表
def read_vocab(vocab_dir):
    with open(vocab_dir, 'r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id
 
 
# 读取分类目录，固定
def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories)))) 
    return categories, cat_to_id
 
 
# 将文件转换为id表示
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])#将每句话id化
        label_id.append(cat_to_id[labels[i]])#每句话对应的类别的id
    
    # # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    #
    return x_pad, y_pad

# TextRNN Model
import torch
from torch import nn
import torch.nn.functional as F
 
# 文本分类，RNN模型
class TextRNN(nn.Module):   
    def __init__(self):
        super(TextRNN, self).__init__()
        # 三个待输入的数据
        self.embedding = nn.Embedding(5000, 64)  # 进行词嵌入
        # self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        self.f1 = nn.Sequential(nn.Linear(256,128),
                                nn.Dropout(0.8),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128,10),
                                nn.Softmax())
 
    def forward(self, x):
        x = self.embedding(x)
        x,_ = self.rnn(x)
        x = F.dropout(x,p=0.8)
        x = self.f1(x[:,-1,:])
        return self.f2(x)


# 获取文本的类别及其对应id的字典
categories, cat_to_id = read_category()
print(categories)
print(cat_to_id)


fpath = open('cnews.vocab.txt', encoding='utf-8', errors='ignore')

#words, word_to_id = read_vocab(fpath) # 获取训练文本中所有出现过的字及其所对应的id
words, word_to_id = read_vocab('cnews.vocab.txt')
vocab_size = len(words) #获取字数
vocab_size2 = len(word_to_id) # 说明有重复文字

print(vocab_size)
print(vocab_size2)

def train():
    model = TextRNN().cuda()
    # 定义损失函数
    Loss = nn.MultiLabelSoftMarginLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # update start
    EPOCH = 20
    costs = []
    early_stop = 0
    min_loss = float('inf')
    best_val_acc = 0.
    # update end

    # 训练
    for epoch in range(1000):
        loses = []
        #print('epoch=', epoch)
        train_correct, train_total = 0., 0.

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x = x_batch.cuda()
            y = y_batch.cuda()
            # 前向传播
            output = model(x)
            loss = Loss(output, y)
            losses.append(loss.item())
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 权值更新
            optimizer.step()

            train_pred = torch.argmax(output, 1).data.cpu().numpy()
            train_label = torch.argmax(b_y, 1).data.cpu().numpy()
            train_correct += (train_pred == train_label).sum()
            train_total += len(train_label)
        meanloss = np.mean(losses)
        costs.append(meanloss)

            #------------------------------------------------------------------------------
            # 计算准确率
            #accuracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).cpu().numpy())
            #print('accuracy:', accuracy)
        # 对模型进行验证
        if epoch % 2 == 0:
            # 训练集准确率
            train_acc = train_correct / train_total
            textrnn.train(False)
            # 验证集预测
            val_correct, val_total = 0., 0.
            for i, (x_v, y_v) in enumerate(valloader):
                x_v, y_v = x_v.to(device), y_v.to(device)
                val_output = textrnn(x_v.to(device))

                # 获取预测的label，并转为数组
                val_pred = torch.argmax(val_output, 1)
                val_pred_arr = val_pred.data.cpu().numpy()
                y_val_arr = torch.argmax(y_v, 1).data.cpu().numpy()
                # 准确率
                val_correct += (val_pred_arr == y_val_arr).sum()
                val_total += len(y_val_arr)
            val_acc = val_correct / val_total
            print("==>epoch:{} 训练集loss:{:.4f} 训练集accuracy:{:.2f}% 验证集accuracy:{:.2f}%".
            format(epoch, meanloss, train_acc * 100, val_acc * 100))
            textrnn.train(True)

            # 根据准确率 保存模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(textrnn, 'textRNN1.pt')
    #  早停法
        if meanloss < min_loss:
            min_loss = meanloss
            early_stop = 0
        else:
            early_stop += 1
        if early_stop > 5:
            print(f"loss连续{epoch}个epoch未降低, 停止循环")
            break

    return model


categories, cat_to_id = read_category() # 获取文本的类别及其对应id的字典
print(categories)

words, word_to_id = read_vocab('cnews.vocab.txt') # 获取训练文本中所有出现过的字及其所对应的id
print(words)


# 获取训练数据每个字的id和对应标签的one-hot形式
x_train, y_train = process_file(train_file, word_to_id, cat_to_id, 100)
print('x_train=', x_train)
x_val, y_val = process_file(val_file, word_to_id, cat_to_id, 100)

# GPU setting
textrnn = TextRNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
textrnn.to(device)

import torch.utils.data as Data
cuda = torch.device('cuda')
x_train, y_train = torch.LongTensor(x_train), torch.Tensor(y_train)
x_val, y_val = torch.LongTensor(x_val), torch.Tensor(y_val)

# train param. setting

train_dataset = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(dataset = train_dataset, batch_size = 88, shuffle = True )

val_dataset = Data.TensorDataset(x_val, y_val)
train_loader = Data.DataLoader(dataset = train_dataset, batch_size = 88)
train()

# CNN模型
# %load cnn_model.py

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据，都是tensor，没有具体的值
        # 建立了三个占位符，此时并没有输入数据，等建立Session,模型开始运行时再通过feed_dict喂入数据。
        # None是bitch_size,input_x是（64，600）的维度，input_y的维度是（64，10）
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        # 指定在第1块gpu上运行，如果指定是cpu则（'/cpu:0'）
        
        with tf.device('/gpu:0'):
            #获取已经存在的变量，并随机初始化，并为之命名。embedding的维度是（5000，64）
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            
            #embedding的过程是，首先拿到一个包含64个样本的batch（64，600），然后取其中一个样本（1，600）
            #然后用每一个字对应的索引取到一个64维的向量，于是每个样本就成了（600，64）的矩阵。
            #tf.nn.embedding_lookup（params, ids）是选取一个张量里面索引对应的元素，
            #如第一个样本是[100,2,..],那么从embedding里取第100行（从0开始计行数）作为第0行，一直这样取600次，再结合要取64个样本
            #所以embedding_inputs.shape=(64，600,64)
            #input_x.shape=(64,600)，如果规定了一批取64个样本。
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            # 使用一维卷积核进行卷积，因为卷积核的第二维与词向量维度相同，只能沿着行向下滑动。
            #对于64个样本中的每一个样本，句子长度为600个字，每个字向量的维度为64，有256个过滤器，卷积核的尺寸为5，
            #那么输入样本为(600,64)经过(5,64)的卷积核卷积后得到(596,1)的向量（600-5+1），默认滑动为1步。
            #由于有256个过滤器，于是得到256个(596,1)的向量。
            #结果显示为(None,596,256)
            #embedding_inputs.shape=(64，600,64)
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            # 用最大池化方法，按行求最大值，conv.shape=[Dimension(None), Dimension(596), Dimension(256)],留下了第1和第3维。
            #取每个向量(596,1)中的最大值，然后就得到了256个最大值，
            #gmp.shape=[Dimension(None), Dimension(256)
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # 神经元的个数为128个，gmp为(64,256),经过这一层得到fc的维度是(64，128）
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            #tf.contrib.layers是比tf.layers封装得更高级的库，这里是进行dropout，
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            #运用RELU激活函数，tf.nn是比tf.layers更底层的库。
            fc = tf.nn.relu(fc)

            # 分类器
            #self.logits的维度是[Dimension(None), Dimension(10)],应该是（64，10）
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            #softmax得到的输出为[Dimension(None), Dimension(10)],是10个类别的概率
            # 然后再从中选出最大的那个值的下标，如[9,1,3...]
            # 最后得到的是（64，1）的列向量，即64个样本对应的类别。
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵，self.logits是全连接层的输出，（64，10），而labels是（64，10）的ont-hot
            # 这个函数先对self.logits进行softmax运算求概率分布，然后再求交叉熵损失
            # 得到的结果维度是（64，1），元素即每个样本对应的交叉熵。
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            # tf.reduce_mean(input_tensor,axis)用于求平均值，这里是求64个样本的交叉熵损失的均值。
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器使用自适应学习率算法。
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率的计算，tf.equal对内部两个向量的每个元素进行对比，返回[True,False,True,...]这样的向量
            # 也就是对预测类别和标签进行对比，self.y_pred_cls形如[9,0,2,3,...]
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            # tf.cast函数将布尔类型转化为浮点型，True转为1.，False转化为0.，返回[1,0,1,...]
            # 然后对[1,0,1,...]这样的向量求均值，恰好就是1的个数除以所有的样本，恰好是准确率。
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
# %load predict.py

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    for i in test_demo:
        print(cnn_model.predict(i))


# %load run_cnn.py
#!/usr/bin/python

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
#模型保存的目录，必须至少有一层文件夹，这里有两层
save_dir = 'checkpoints/textcnn'
#这里说是保存路径，其实这个“best_validation”是保存的文件的名字的开头，比如保存的一个文件是“best_validation.index”
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    #round函数是对浮点数四舍五入为int，注意版本3中round(0.5)=0,round(3.567,2)=3.57。
    #timedelta是用于对间隔进行规范化输出，间隔10秒的输出为：00:00:10
    
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_) #test_data有10000个样本
    # 生成一个迭代器，每次循环都会得到两个矩阵，分别是x_batch和y_batch。
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch) #128，这里一个批量是取128个样本
        
        #1.0是dropout值，在测试时不需要舍弃，feed_dict得到一个字典，包含x,y和keep_drop值。
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        #把feed_dict的数据传入去计算model.loss,是求出了128个样本的平均交叉熵损失
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        # 把平均交叉熵和平均准确率分别乘以128个样本得到总数，在不断累加得到10000个样本的总数。
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    # 求出10000个样本的平均交叉熵，和平均准确率。
    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    
    #获取准备数据所花费的时间
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    # tf.global_variables_initializer()添加节点用于初始化所有的变量，为必备语句
    session.run(tf.global_variables_initializer())
    # 和writer = tf.summary.Filewriter()搭配食用生成神经网络数据流程图，为必备语句

    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    # epoch=10,训练10轮
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        #batch_train是一个能返回两个矩阵的迭代器
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        
        # x_batch.shape为(64,600)
        for x_batch, y_batch in batch_train:
            #构造一个字典，以满足神经网络的输入需要
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            #print("x_batch is {}".format(x_batch.shape))
            # total_batch初始值为0，save.per_batch=10
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                #下面这个语句是难点，后续要继续研究。
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    # 下面两个的维度分别为(10000,600),(10000,10)
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
    
    
    # saver的操作必须在sess建立后进行。
    session = tf.Session()
    # 这里为啥要初始化呢，可能是前面定义了变量，比如用tf.placeholder,后面再看看。
    session.run(tf.global_variables_initializer())
    # 在保存和恢复模型时都需要首先运行这一行：tf.train.Saver()，而不是只有保存时需要。
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    # 返回了10000个总测试样本的平均交叉熵损失和平均准率。
    loss_test, acc_test = evaluate(session, x_test, y_test)
    
    # 格式化输出，6表示分配6个占位符，>表示对传入的值右对齐，2表示保留2位小数。
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test) #为10000
    num_batch = int((data_len - 1) / batch_size) + 1 #为79，也就是1轮下来要取79个批量
    # 得到[0,0,..,6,6,6,...,9,9,9]这样的向量，也就是每个样本的标签转化为对应的数字索引。
    y_test_cls = np.argmax(y_test, 1)
    #生成一个(10000,)的数组，用来保存预测值。
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        # i=0时，取[0,1*128),i=77时，取[77*128,78*128),i=78时，取[78*128,10000),因为79*128=10112>10000.
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0 #测试的时候不需要dropout神经元。
        }
        #等号右边得到是（128，）的数组，即每个样本的类别，[0,0,...,6,6,6,..]这样的。
        # y_pred_cls仍然是（10000，）的数组。
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    # 可以得到准确率 、找回率和F1_score
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)
    option='test'
    if option == 'train':
        train()
    else:
        test()
        
  # %load cnews_loader.py

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    #长度为50000
    data_len = len(x)
    #int()可以将其他类型转化为整型，也可以用于向下取整，这里为782.
    num_batch = int((data_len - 1) / batch_size) + 1
    #元素的范围是0-49999，形如[256,189,2,...]的拥有50000个元素的列表
    indices = np.random.permutation(np.arange(data_len))
    # 用indices对样本和标签按照行进行重新洗牌，接着上面的例子，把第256行(从0开始计)放在第0行，第189行放在第1行.
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        #i=780时，end_id=781*64=49984;
        #当i=781时，end_id=50000，因为782*64=50048>50000,所以最后一批取[49984:50000]
        end_id = min((i + 1) * batch_size, data_len)
        # yield是生成一个迭代器，用for循环来不断生成下一个批量。
        # 为了防止内存溢出，每次只取64个，内存占用少。
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]



