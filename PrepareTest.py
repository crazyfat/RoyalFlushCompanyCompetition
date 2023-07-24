from torch.utils.data import Dataset
import pandas as pd
import torch


class TestDataProcess(Dataset):

    def __init__(self, bert_tokenizer, file_path):
        self.bert_tokenizer = bert_tokenizer
        self.inputs_idx, self.token_type_ids, self.attention_mask = self.get_data(file_path)

    def __len__(self):
        return len(self.inputs_idx)

    def __getitem__(self, idx):
        return self.inputs_idx[idx], self.token_type_ids[idx], self.attention_mask[idx]


    def get_data(self, file):

        df = pd.read_csv(file, sep='\t', header=None)
        seq = zip(df[0], df[1])
        data = self.bert_tokenizer.batch_encode_plus(seq, padding=True, truncation=False, return_tensors='pt')
        inputs_idx = data['input_ids']
        token_type_ids = data['token_type_ids']
        attention_mask = data['attention_mask']
        return inputs_idx, token_type_ids, attention_mask

        # 接下来倒序生成样本
        # re_seq = zip(df[2],df[1])
        # re_data = self.bert_tokenizer.batch_encode_plus(re_seq, padding=True, truncation=False, return_tensors='py')
