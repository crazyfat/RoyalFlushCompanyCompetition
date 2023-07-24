from bert import mybert
from transformers import BertTokenizer, AdamW
import torch
from utils import trainBert, testBert
from PrepareData import DataProcess
from PrepareTest import TestDataProcess
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

EPOCH = 5
batch_size = 128
lr = 2e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('./bert/vocab.txt')

print("====loading data====")
train_data = DataProcess(bert_tokenizer, './data/train.tsv')
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
print("====loading data finish====")

# ======训练测试======
test_data = TestDataProcess(bert_tokenizer, './data/test.tsv')
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
# ======训练测试======

my_model = mybert().to(device)
param_optimizer = list(my_model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {
            'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':0.01
        },
        {
            'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':0.0
        }
]
opts = AdamW(optimizer_grouped_parameters, lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opts, mode="max",
                                                           factor=0.85, patience=0)
print("====begin training====")

trainBert(my_model, opts, EPOCH, train_loader)

print("====begin testing====")

testBert(my_model, test_loader)
