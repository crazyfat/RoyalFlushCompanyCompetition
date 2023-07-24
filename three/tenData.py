from torch.utils.data import Dataset
import pandas as pd
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
        labels = df[0][:100].values
        seq = zip(df[1][:100], df[2][:100])
        data = self.bert_tokenizer.batch_encode_plus(seq, padding=True, truncation=False, return_tensors='pt')
        inputs_idx = data['input_ids']
        token_type_ids = data['token_type_ids']
        attention_mask = data['attention_mask']
        return inputs_idx, token_type_ids, attention_mask, torch.Tensor(labels).type(torch.long)