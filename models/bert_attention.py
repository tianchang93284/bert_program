# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer, tokenization
import math

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert_RNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt',encoding='utf-8').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.output_path = 'output/ChnSentiCorp'                                      # 输出路径
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 30                                             # epoch数
        self.batch_size = 16                                           # mini-batch大小
        self.pad_size = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './pretrained_models/bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (3, 5, 10)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.5
        self.rnn_hidden = 768
        self.trunc_medium = -1
        self.num_layers = 2
        self.threshold = 0.5

class LabelLstmOutPut():
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addLabel(self, label):
        for word in label:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.index2word[self.n_words] += 1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * config.num_classes * 2, config.num_classes)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.label_embedding = nn.Embedding(50, 256)
        self.label_lstm = nn.LSTM(256, 768, num_layers=1, bidirectional=False)
        # hidden_param = nn.Parameter(torch.Tensor(768, 768))
        label_list = ["政府采购", "科技基础设施建设", "市场监管", "公共服务", "科技成果转移转化",
                           "科创基地与平台", "金融支持", "教育和科普", "人才队伍", "贸易协定", "税收激励",
                           "创造和知识产权保护", "项目计划", "财政支持", "技术研发"
                           ]
        self.label_ids = []
        for item in label_list:
            item = ['CLS'] + list(item) + ['CLS']
            ids = config.tokenizer.convert_tokens_to_ids(item)
            self.label_ids.append(ids)
    def attention_net(self, x):  # x:[batch, seq_len, hidden_dim*2]
        u = torch.tanh(torch.matmul(x, self.w_omega))  # [batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = self.sigmod(att, dim=1)
        scored_x = x * att_score  # [batch, seq_len, hidden_dim*2]
        context = torch.sum(scored_x, dim=1)  # [batch, hidden_dim*2]
        return context

    def soft_attention_net(self, x, query, mask=None):  # 软性注意力机制（key=value=x）

        d_k = query.size(-1)  # d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # 打分机制  scores:[batch, seq_len, seq_len]
        p_attn = self.sigmod(scores, dim=-1)  # 对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)  # 对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]

        return context, p_attn

    def label_lstm_out(self):
        # label_list = ["政府采购", "科技基础设施建设", "市场监管", "公共服务", "科技成果转移转化",
        #               "科创基地与平台", "金融支持", "教育和科普", "人才队伍", "贸易协定", "税收激励",
        #               "创造和知识产权保护", "项目计划", "财政支持", "技术研发"
        #               ]
        #
        # embedding = nn.Embedding(50, 256)
        # lstm = nn.LSTM(256, 768, num_layers=1, bidirectional=False)
        # #hidden_param = nn.Parameter(torch.Tensor(768, 768))
        # output_lstm_label = []
        # for item in label_list:
        #     item = ['CLS'] + list(item) + ['CLS']
        #     ids = self.config.tokenizer.convert_tokens_to_ids(item)
        #     output, (lstmhidden, c_out) = lstm(embedding(ids))
        #     output_lstm_label.append(lstmhidden)
        # return output_lstm_label
        pass

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)

        output_lstm_label = []
        for item in self.label_ids:
            output, (lstmhidden, c_out) = self.label_lstm(self.label_embedding(item))
            output_lstm_label.append(lstmhidden)

        temp_text_cls = text_cls.clone()
        for i, label_hidden in enumerate(output_lstm_label):
            if i == 0:
                text_cls = torch.cat((text_cls, label_hidden))
            else:
                text_cls = torch.cat([text_cls, temp_text_cls, label_hidden])
        context = self.attention_net(text_cls)
        out = self.dropout(context)
        out = self.fc_rnn(out)  # 句子最后时刻的 hidden state
        return out
