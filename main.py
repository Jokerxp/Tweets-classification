# Imports
import os
import time
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models
import argparse

# Global definitions - data
DATA_FN = 'data/crowdflower_data.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()
# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator, epochs, scheduler=None,
                early_stopping=False):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """

    stop_sign = False
    best = float("inf")
    device = "cuda"
    if USE_CUDA:
        model.to(device)
    model.train()
    start = time.time()
    for epoch in range(epochs):

        train_loss_sum, train_acc_sum, n = 0, 0, 0
        for X, y in train_generator:
            # print(X.shape,y.shape)

            if USE_CUDA:
                X = X.to(device)
                y = y.to(device=device, dtype=torch.long)
            else:
                y = y.long()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).cpu().sum().item()
            n += y.shape[0]
        if scheduler is not None:
            # scheduler.step()
            # print(scheduler.get_lr())
            scheduler(optimizer, epoch)
            print("learning rate: ", optimizer.param_groups[0]["lr"])

        dev_acc, dev_loss = models.evaluate_dev(dev_generator, model, loss_fn, USE_CUDA)
        # Early stopping for extension2 Extension-grading
        if early_stopping:
            if dev_loss > best - 0.01:
                if stop_sign is False:
                    stop_sign = True
                else:
                    print("Early Stopping")
                    break
            elif best - 0.01 < dev_loss <= best:
                best = dev_loss
            else:
                stop_sign = False
                best = dev_loss
        print(
            'epoch {:d}, development loss {:.3f}'.format(
                epoch + 1, dev_loss))
    end = time.time()
    print("Total time {:.2f}".format(end - start))
    return model


def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            if USE_CUDA:
                X_b = X_b.to("cuda")
                y_b = y_b.to(device="cuda",dtype=torch.long)
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                      BATCH_SIZE,
                                                                                                      EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    ########## YOUR CODE HERE ##########
    # TODO: for each of the two models, you should 1) create it,
    # TODO 2) run train_model() to train it, and
    # TODO: 3) run test_model() on the result
    model_type = list(vars(args).values())[0]
    model, lr, epochs = None, None, 5
    print("Train on {}:".format("cuda" if USE_CUDA else "cpu"))
    if model_type == "dense":
        model = models.DenseNetwork(embeddings, USE_CUDA, nums_hidden1=64, nums_hidden2=128, nums_classes=4,
                                    embedding_size=100)
        lr = 0.001
        epochs = 20
    if model_type == "RNN":
        model = models.RecurrentNetwork(embeddings, num_hiddens=50, USE_CUDA=USE_CUDA, num_layers=2,
                                        embedding_size=100, num_steps=91, num_classes=4)
        lr = 0.005
        epochs = 15
    # Extension1 CNN model Extension-grading
    if model_type == "extension1":
        print("This is a CNN model.")
        model = models.ExperimentalNetwork(embeddings, USE_CUDA, out_channels=128, num_classes=4, kernel_range=(1, 4),
                                           embedding_size=100)
        lr = 0.0005
        epochs = 5
    # Extension2  learning rate scheduler and early stopping
    if model_type == "extension2":
        print(
            "This is a Dense network with batch normalization, a learning rate scheduler and early stopping.")
        model = models.DenseNetwork2(embeddings, USE_CUDA, nums_hidden1=128, nums_hidden2=256, nums_classes=4,
                                     embedding_size=100)
        lr = 0.005
        epochs = 10

    if model is not None and lr is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        early_stopping = False
        if model_type == "extension2":
            scheduler = models.lrscheduler
            early_stopping = True
        model = train_model(model, loss_fn, optimizer, train_generator, dev_generator, epochs, scheduler,
                            early_stopping)
        test_model(model, loss_fn, test_generator)
    else:
        raise NameError("Please input correct arguments")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
