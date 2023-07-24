from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class DataProcess(Dataset):

    def __init__(self, bert_tokenizer, file_path):
        self.bert_tokenizer = bert_tokenizer
        self.inputs_idx, self.token_type_ids, self.attention_mask, self.labels = self.get_data(file_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs_idx[idx], self.token_type_ids[idx], self.attention_mask[idx], self.labels[idx]

    def get_data(self, file):

        df = pd.read_csv(file, sep='\t', header=None)
        labels = df[0].values
        seq = zip(df[1], df[2])
        data = self.bert_tokenizer.batch_encode_plus(seq, padding=True, truncation=False, return_tensors='pt')
        inputs_idx = data['input_ids']
        token_type_ids = data['token_type_ids']
        attention_mask = data['attention_mask']

        # # 接下来倒序生成样本
        # re_seq = zip(df[2], df[1])
        # re_data = self.bert_tokenizer.batch_encode_plus(re_seq, padding=True, truncation=False, return_tensors='pt')
        # re_input_idx = re_data['input_ids']
        # re_token_idx = re_data['token_type_ids']
        # re_attention_mask = data['attention_mask']
        # inputs_idx = torch.cat((inputs_idx, re_input_idx), 0)
        # token_type_ids = torch.cat((token_type_ids, re_token_idx), 0)
        # attention_mask = torch.cat((attention_mask, re_attention_mask), 0)
        # b = np.concatenate((labels, labels))

        return inputs_idx, token_type_ids, attention_mask, torch.Tensor(labels).type(torch.long)