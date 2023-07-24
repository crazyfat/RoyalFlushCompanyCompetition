import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from bert import BertModel
from utils import test
from data import DataPrecessForSentence
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('./bert/vocab.txt')
print("=====loading test data====")
train_data = DataProcess(bert_tokenizer, './data/test.tsv')
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
print("====loading data finish====")