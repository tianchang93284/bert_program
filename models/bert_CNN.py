# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert_CNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt',encoding='utf-8').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.output_path = 'output/ChnSentiCorp'                                     # 输出路径
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 30                                             # epoch数
        self.batch_size = 4                                           # mini-batch大小
        self.pad_size = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 3e-5                                       # 学习率
        self.bert_path = './pretrained_models/bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.trunc_medium = 0
        self.filter_sizes = (3, 5, 10)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.5
        self.threshold = 0.5

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.conv1 = nn.Conv2d(1, 256, (1, 768))
        self.conv1_1 = nn.Conv2d(256, 256, (1, 1))
        self.conv2 = nn.Conv2d(1, 256, (3, 768), padding=(1,1))
        self.conv3 = nn.Conv2d(256, 256, (5, 1), padding=(2, 2))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(256)
        self.fc = nn.Linear(config.num_filters * 4, config.num_classes)
        #self.maxpool = nn.MaxPool2d()
        #self.convs = nn.ModuleList(
            #[nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        #self.dropout = nn.Dropout(config.dropout)

        #self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def max_pool(self, x):
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        out_1 = self.conv1(out)
        out_2 = self.conv1(out)
        out_3 = self.conv2(out)
        out_4 = self.conv2(out)

        out_2 = self.bn(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.conv1_1(out_2)

        out_3 = self.bn(out_3)
        out_3 = self.relu(out_3)
        out_3 = self.conv3(out_3)

        out = torch.cat([self.max_pool(out_1), self.max_pool(out_2), self.max_pool(out_3), self.max_pool(out_4)], 1).squeeze(0)
        out = self.fc(out)
        #out = self.bn(out)
        #out = self.relu(out)

        # out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # out = self.dropout(out)
        # out = self.fc_cnn(out)
        return out
