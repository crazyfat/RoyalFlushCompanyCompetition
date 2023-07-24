import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification

path = './data'
bert_path = './bert/bert-base-chinese'

class mybert(nn.Module):
    def __init__(self):
        super(mybert, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("./bert/bert-base-chinese",num_labels = 2)
        self.device = torch.device("cuda:2")
        for param in self.bert.parameters():
            param.requires_grad = True
    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments,labels):
        output = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks,token_type_ids=batch_seq_segments, labels = labels)
        loss = output.loss
        logits = output.logits
        probabilities = F.softmax(logits)
        return loss,logits,probabilities