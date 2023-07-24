import torch
import torch.nn as nn
import time
from tqdm import tqdm
from utils.metric import correct_predictions


def train(model, opts, EPOCH, train_loader=None):
    train_loss = []
    for epoch in range(0, EPOCH):
        print("epoch {}:".format(epoch))
        model.train()
        device = model.device
        epoch_start = time.time()
        batch_time_avg = 0.0
        running_loss = 0.0
        correct_preds = 0
        tqdm_batch_iterator = tqdm(train_loader)
        for batch_index, (batch_seqs, batch_seq_segments, batch_seq_masks, batch_labels) in enumerate(
                tqdm_batch_iterator):
            batch_start = time.time()
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(
                device), batch_labels.to(device)
            opts.zero_grad()
            loss, logits, probabilities = model(seqs, masks, segments, labels)
            loss = loss.to(torch.float32)
            loss.backward()
            opts.step()
            batch_time_avg += time.time() - batch_start
            running_loss += loss.item()
            correct_preds += correct_predictions(probabilities, labels)
            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
                .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / len(train_loader.dataset)
        train_loss.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
    torch.save(model, 'net.pkl')