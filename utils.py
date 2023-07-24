import torch
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score,accuracy_score


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)      # 预测答案正确
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)      #
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)        # 错将其他类预测为本类
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)        # 本类标签预测为其他类标

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def trainBert(model, opts, EPOCH, train_loader):
    train_loss = []
    for epoch in range(0, EPOCH):
        print("epoch {}:".format(epoch))
        model.train()
        device = model.device
        epoch_start = time.time()
        batch_time_avg = 0.0
        running_loss = 0.0
        correct_preds = 0
        batch_f1 = 0.0
        tqdm_batch_iterator = tqdm(train_loader)
        for batch_index, (batch_seqs, batch_seq_segments, batch_seq_masks, batch_labels) in enumerate(tqdm_batch_iterator):
            batch_start = time.time()
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(
                device), batch_labels.to(device)
            opts.zero_grad()
            loss, logits, probabilities = model(seqs, masks, segments, labels)
            loss = loss.to(torch.float32)
            loss.backward()
            opts.step()
            # if preds == None:
            #     preds = logits.detach().cpu.numpy()
            #     target_labels = labels.detach().cpu.numpy()
            # else:
            #     preds = np.append(preds, logits.detach().cpu.numpy(), axis=0)
            #     target_labels = np.append(target_labels, labels.detach().cpu.numpy(), axis=0)
            batch_time_avg += time.time() - batch_start
            running_loss += loss.item()
            correct_preds += correct_predictions(probabilities, labels)
            _, preds = probabilities.max(dim=1)
            preds = preds.cpu().numpy()
            la = labels.cpu().numpy()
            score = f1_score(la, preds)
            score = score.item()
            batch_f1 = batch_f1+score
            description = "Avg.batch.time: {:.4f}s, loss: {:.4f}, f1: {:.4f}%" \
                .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1), score*100)
            tqdm_batch_iterator.set_description(description)
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / len(train_loader.dataset)
        epoch_f1 = batch_f1 / len(train_loader)
        train_loss.append(epoch_loss)
        print("time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%, f1_score: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_f1*100))

    torch.save(model.state_dict(), 'parameter.pkl')

def testBert(model, test_loader):
    model.eval()
    device = model.device
    labels = []
    with torch.no_grad():
        for batch_index, (batch_seqs, batch_seq_segments, batch_seq_masks) in enumerate(test_loader):
            seqs, masks, segments = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(
                device)
            _, _, probabilities = model(seqs, masks, segments, labels = None)
            _, preds = probabilities.max(dim = 1)
            label = preds.cpu().numpy()
            labels.extend(label)
    print("=====writing file=====")
    x = ""
    for i in range(len(labels)):
        x += "{\"label\": " + str(labels[i]) + "}\n"
    with open('./data/testx.json', 'w') as f:
        f.write(x)
        # 记录到3