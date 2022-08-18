

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np


class DenseNetwork(nn.Module):
    def __init__(self, embeddings, USE_CUDA, nums_hidden1=32, nums_hidden2=64, nums_classes=4, embedding_size=100):
        torch.cuda.manual_seed_all(10)
        torch.manual_seed(0)
        super(DenseNetwork, self).__init__()
        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.embeddings = embeddings
        self.USE_CUDA = USE_CUDA
        self.nums_hidden1 = nums_hidden1
        self.nums_hidden2 = nums_hidden2
        self.num_classes = nums_classes

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings)
        self.dnn = nn.Sequential(
            nn.Linear(embedding_size, nums_hidden1),
            # nn.Dropout(0.3),
            nn.Tanh(),

            nn.Linear(nums_hidden1, nums_hidden2),
            # nn.Dropout(0.3),
            nn.Tanh(),

            nn.Linear(nums_hidden2, nums_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        embeded = self.embedding_layer(x).float()
        embeded = nn.AvgPool2d((embeded.shape[1], 1), 1)(embeded).squeeze()
        output = self.dnn(embeded)

        return output


class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings, num_hiddens, USE_CUDA=True, num_layers=2, embedding_size=100, num_steps=91,
                 num_classes=4):
        super(RecurrentNetwork, self).__init__()

        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.embeddings = embeddings
        self.USE_CUDA = USE_CUDA
        self.num_steps = num_steps

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings)
        self.LSTM = nn.LSTM(embedding_size, num_hiddens, num_layers)
        self.fc = nn.Linear(num_hiddens, num_classes)

        for name,param in self.LSTM.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
    # x is a PaddedSequence for an RNN
    def forward(self, x):
        embeded = self.embedding_layer(x.transpose(0, 1)).float()
        y, (h,_) = self.LSTM(embeded)
        # print(h[-1,:,:])
        output = self.fc(y[-1,:,:])

        return output

# This is a CNN model
# Extension-grading
class ExperimentalNetwork(nn.Module):
    def __init__(self, embeddings, USE_CUDA=True, out_channels=128, num_classes=4, kernel_range=(1, 4),
                 embedding_size=100):
        super(ExperimentalNetwork, self).__init__()
        ########## YOUR CODE HERE ##########
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.kernel_range = kernel_range
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.USE_CUDA = USE_CUDA

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings)

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, out_channels=out_channels, kernel_size=(k,)),
            nn.ReLU(),
            # nn.MaxPool1d(91-k+1)
        ) for k in range(kernel_range[0], kernel_range[1] + 1)])

        self.fc = nn.Linear((kernel_range[1] - kernel_range[0] + 1) * out_channels, num_classes)

    # x is a PaddedSequence
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        embeded = self.embedding_layer(x).float()
        embeded = embeded.permute(0, 2, 1)
        output = [torch.max(conv(embeded), dim=2).values for conv in self.convs]
        y = torch.cat(output, dim=1)
        y = self.fc(y)

        return y


# This is a densenetwork for extension2
# Extension-grading
class DenseNetwork2(nn.Module):
    def __init__(self, embeddings, USE_CUDA, nums_hidden1=32, nums_hidden2=64, nums_classes=4, embedding_size=100):
        super(DenseNetwork2, self).__init__()
        ########## YOUR CODE HERE ##########
        self.embeddings = embeddings
        self.USE_CUDA = USE_CUDA
        self.nums_hidden1 = nums_hidden1
        self.nums_hidden2 = nums_hidden2
        self.num_classes = nums_classes

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings)
        self.dnn = nn.Sequential(
            nn.Linear(embedding_size, nums_hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(nums_hidden1),

            nn.Linear(nums_hidden1, nums_hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(nums_hidden2),

            nn.Linear(nums_hidden2, nums_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        embeded = self.embedding_layer(x).float()
        embeded = nn.AvgPool2d((embeded.shape[1], 1), 1)(embeded).squeeze()
        output = self.dnn(embeded)
        return output


def evaluate_dev(dev_generator, model, loss_fn, USE_CUDA):
    dev_loss_sum, dev_acc_sum, n = 0, 0, 0
    for X, y in dev_generator:
        if USE_CUDA:
            X = X.to("cuda")
            y = y.to(device="cuda", dtype=torch.long)
        else:
            y = y.long()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        dev_loss_sum += loss.item()
        dev_acc_sum += (y_hat.argmax(dim=1) == y).cpu().sum().item()
        n += y.shape[0]
    return dev_acc_sum / n, dev_loss_sum


# Learning rate scheduler for extension2 Extension-grading
def lrscheduler(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    if (epoch + 1) % 3 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
