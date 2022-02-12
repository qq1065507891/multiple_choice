import torch.nn as nn
import torch

from transformers import BertModel


class MultipleChoice(nn.Module):
    def __init__(self, config):
        super(MultipleChoice, self).__init__()
        self.config = config

        self.model_name = config.model_name

        self.bert = BertModel.from_pretrained(config.bert_base_path)

        self.fc = nn.Linear(self.bert.config.hidden_size, self.config.hidden_size)

        self.drop_out = nn.Dropout(self.config.drop_out)

        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, inputs):
        """
        :param inputs: 【input_ids, attention_mask, attention_mask】
        :return:
        """
        input_ids, token_type_ids, attention_mask = inputs

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :]

        drop_out = self.drop_out(pooled_output)
        logits = self.classifier(drop_out)
        reshaped_logits = logits.view(-1, self.config.num_choices)

        return reshaped_logits


class bert_lstm(nn.Module):
    def __init__(self, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super(bert_lstm, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.bert = BertModel.from_pretrained("../chinese-bert_chinese_wwm_pytorch/data")
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        # self.sig = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        # 生成bert字向量
        x = self.bert(x)[0]  # bert 字向量

        # lstm_out
        # x = x.float()
        lstm_out, (hidden_last, cn_last) = self.lstm(x)
        # print(lstm_out.shape)   #[32,100,768]
        # print(hidden_last.shape)   #[4, 32, 384]
        # print(cn_last.shape)    #[4, 32, 384]

        # 修改 双向的需要单独处理
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]  # [32, 384]

        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        # print(out.shape)    #[32,768]
        out = self.fc(out)

        return out

