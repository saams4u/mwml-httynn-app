# train.py - initialize and train a model.

import os
import sys

sys.path.append(".")

from argparse import ArgumentParser
from datetime import datetime

import io
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import random
import time

from tqdm import tqdm

import wandb

wandb.init(project="mwml-httynn-app_v2")

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_classification import config, data, models, utils


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()

    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train_step(model, optimizer, dataloader, device):
    """Train step."""
    # Set model to train mode
    model.train()
    train_loss, correct = 0., 0

    # Iterate over train batches
    num_batches = len(dataloader)

    for i, (X, y) in tqdm(enumerate(dataloader), total=num_batches):

        # Step
        X, y = X.to(device), y.to(device)  # Set device
        optimizer.zero_grad()  # Reset gradients
        _, logits = model(X)  # Forward pass
        
        loss = F.cross_entropy(logits, y)  # Define loss
        # loss = F.binary_cross_entropy_with_logits(logits, y)  # Define loss

        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Metrics
        y_pred = logits.max(dim=1)[1]
        # y_pred = torch.round(torch.sigmoid(logits[1]))

        correct += torch.eq(y_pred, y).sum().item()
        train_loss += (loss.detach().item() - train_loss) / (i + 1)

    train_acc = 100. * correct / len(dataloader.dataset)

    return train_loss, train_acc


def test_step(model, dataloader, device):
    """Validation or test step."""
    # Set model to eval mode
    model.eval()

    loss, correct = 0., 0
    y_preds, y_targets = [], []

    # Iterate over val batches
    num_batches = len(dataloader)

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):

            # Step
            X, y = X.to(device), y.to(device)  # Set device
            _, logits = model(X)  # Forward pass

            # Metrics
            loss += F.cross_entropy(logits, y, reduction='sum').item()
            # loss += F.binary_cross_entropy_with_logits(logits, y, reduction='sum')  # Define loss

            y_pred = logits.max(dim=1)[1]
            correct += torch.eq(y_pred, y).sum().item()

            # Store outputs
            y_preds.extend(y_pred.cpu().numpy())
            y_targets.extend(y.cpu().numpy())

    loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)

    return loss, accuracy, y_preds, y_targets


def train(model, optimizer, scheduler, num_epochs, patience,
          train_dataloader, val_dataloader, test_dataloader, device):

    best_val_loss = np.inf
    config.logger.info("Training:")

    for epoch in range(num_epochs):
        # Steps
        train_loss, train_acc = train_step(model, optimizer, train_dataloader, device)
        test_loss, test_acc, _, _ = test_step(model, val_dataloader, device)

        # print(f'Epoch {epoch+0:03}: | Train Loss: {train_loss / len(train_dataloader):.5f} | Train Acc: {train_acc} | Val Loss: {test_loss / len(val_dataloader):.5f} | Val Acc: {test_acc}')

        # Adjust learning rate
        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            _patience = patience  # reset _patience
            # Metrics
            config.logger.info(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.2f}, train_acc: {train_acc:.1f}, "
                f"val_loss: {test_loss:.2f}, val_acc: {test_acc:.1f}")
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": test_loss,
                "val_accuracy": test_acc})
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        else:
            _patience -= 1
        if not _patience:  # 0
            config.logger.info("Stopping early!")
            break

    return best_val_loss


def plot_confusion_matrix(y_pred, y_target, classes, fp, cmap=plt.cm.Blues):
    """Plot a confusion matrix using ground truth and predictions."""
    # Confusion matrix
    cm = np.round(confusion_matrix(y_target, y_pred))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #  Figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Axis
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)

    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Values
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:d} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Save
    plt.rcParams["figure.figsize"] = (7, 7)
    plt.savefig(fp)


def get_performance(y_pred, y_target, classes):
    """Per-class performance metrics. """
    performance = {'overall': {}, 'class': {}}
    metrics = np.round(precision_recall_fscore_support(y_target, y_pred))

    # Overall performance
    performance['overall']['precision'] = np.mean(metrics[0])
    performance['overall']['recall'] = np.mean(metrics[1])
    performance['overall']['f1'] = np.mean(metrics[2])
    performance['overall']['num_samples'] = np.float64(np.sum(metrics[3]))

    # Per-class performance
    for i in range(len(classes)):
        performance['class'][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i])
        }

    return performance


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        default=False, help="Use GPUs")
    parser.add_argument('--seed', type=int, default=1234,
                        help="initialization seed")
    parser.add_argument('--shuffle', action='store_true',
                        default=False, help="shuffle your data")
    parser.add_argument('--data-url', type=str,
                        required=True, help="URL of data file")
    parser.add_argument('--lower', action='store_true',
                        default=False, help="lowercase all text")
    parser.add_argument('--binary', action='store_true',
                        default=False, help="set to binary classification")
    parser.add_argument('--char-level', action='store_true',
                        default=False, help="split text to character level")
    parser.add_argument('--filters', type=str,
                        default=r"[!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~]",
                        help="text preprocessing filters")
    parser.add_argument('--data-size', type=float,
                        default=1.0, help="proportion of data to use")
    parser.add_argument('--train-size', type=float,
                        default=0.7, help="train data proportion")
    parser.add_argument('--val-size', type=float,
                        default=0.15, help="val data proportion")
    parser.add_argument('--test-size', type=float,
                        default=0.15, help="test data proportion")
    parser.add_argument('--num-epochs', type=int,
                        default=100, help="# of epochs to train")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="# of samples per batch")
    parser.add_argument('--embedding-dim', type=int,
                        default=100,
                        help="dimension of embeddings (50, 100, 200, 300 if using GloVe)")
    parser.add_argument('--use-glove', action='store_true',
                        default=False, help="Use pretrained GloVe embeddings")
    parser.add_argument('--freeze-embeddings', action='store_true',
                        default=False, help="Freeze embeddings during training")
    parser.add_argument('--filter-sizes', nargs='+',
                        default=[2, 3, 4], type=int, help="cnn filter sizes")
    parser.add_argument('--num-filters', type=int, default=50,
                        help="# of filters per cnn filter size")
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help="# of hidden units in fc dense layers")
    # parser.add_argument('--stacked-layers', type=int, default=8,
    #                     help="# of stacked lstm layers")
    parser.add_argument('--dropout-p', type=float, default=0.2,
                        help="dropout proportion in fc dense layers")
    parser.add_argument('--learning-rate', type=float,
                        default=1e-4, help="initial learning rate")
    parser.add_argument('--patience', type=int, default=3,
                        help="# of epochs of continued performance regression")
    
    args = parser.parse_args()
    config.logger.info(json.dumps(args.__dict__, indent=2))
    
    wandb.config.update(args)
    wandb.log({"run_dir": wandb.run.dir})

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # multi-GPU.
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set device
    device = torch.device('cuda' if (
        torch.cuda.is_available() and args.cuda) else 'cpu')

    # Load data
    X, y = data.load_data(url=args.data_url, data_size=args.data_size)
    config.logger.info(
        "Raw data:\n"
        f"  {X[0]} → {y[0]}")

    # Preprocesss
    original_X = X
    X = data.preprocess_texts(texts=X, binary=args.binary, 
                              lower=args.lower, filters=args.filters)
    
    config.logger.info(
        "Preprocessed data:\n"
        f"  {original_X[0]} → {X[0]}")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data.train_val_test_split(
        X=X, y=y, val_size=args.val_size, test_size=args.test_size, shuffle=args.shuffle)
    
    config.logger.info(
        "Data splits:\n"
        f"  X_train: {len(X_train)}, y_train: {len(y_train)}\n"
        f"  X_val: {len(X_val)}, y_val: {len(y_val)}\n"
        f"  X_test: {len(X_test)}, y_test: {len(y_test)}")

    # Tokenizer
    X_tokenizer = data.Tokenizer(char_level=args.char_level)
    X_tokenizer.fit_on_texts(texts=X_train)
    
    vocab_size = len(X_tokenizer) + 1
    
    config.logger.info(
        "X tokenizer:\n"
        f"  {X_tokenizer}")

    # Convert texts to sequences of indices
    original_text = X_train[0]
    X_train = np.array(X_tokenizer.texts_to_sequences(X_train))
    X_val = np.array(X_tokenizer.texts_to_sequences(X_val))
    X_test = np.array(X_tokenizer.texts_to_sequences(X_test))
    preprocessed_text = X_tokenizer.sequences_to_texts([X_train[0]])[0]
    
    config.logger.info(
        "Text to indices:\n"
        f"  (raw) {original_text}\n"
        f"  (preprocessed) → {preprocessed_text}\n"
        f"  (tokenized) → {X_train[0]}")

    # Label encoder
    y_tokenizer = data.LabelEncoder()
    y_tokenizer = y_tokenizer.fit(y_train)
    
    classes = y_tokenizer.classes
    
    config.logger.info(
        "y tokenizer:\n"
        f"  {y_tokenizer}\n"
        "classes:\n"
        f"  {classes}")

    # Convert labels to tokens
    class_ = y_train[0]

    y_train = y_tokenizer.transform(y_train)
    y_val = y_tokenizer.transform(y_val)
    y_test = y_tokenizer.transform(y_test)
    
    config.logger.info(
        "Labels to indices:\n"
        f"  {class_} → {y_train[0]}")

    # Class weights
    counts = np.bincount(y_train)
    class_weights = {i: 1.0/count for i, count in enumerate(counts)}
    
    config.logger.info(
        "class counts:\n"
        f"  {counts}\n"
        "class weights:\n"
        f"  {class_weights}")

    # Create datasets
    # For ModelLSTM architecture
    # train_dataset = data.Model_LSTM_Dataset(X=X_train, y=y_train)
    # val_dataset = data.Model_LSTM_Dataset(X=X_val, y=y_val)
    # test_dataset = data.Model_LSTM_Dataset(X=X_test, y=y_test)

    # // For TextCNN architecture
    train_dataset = data.Text_CNN_Dataset(
        X=X_train, y=y_train, max_filter_size=max(args.filter_sizes))
    val_dataset = data.Text_CNN_Dataset(
        X=X_val, y=y_val, max_filter_size=max(args.filter_sizes))
    test_dataset = data.Text_CNN_Dataset(
        X=X_test, y=y_test, max_filter_size=max(args.filter_sizes))
    
    config.logger.info(
        "Data splits:\n"
        f"  Train dataset:{train_dataset.__str__()}\n"
        f"  Val dataset: {val_dataset.__str__()}\n"
        f"  Test dataset: {test_dataset.__str__()}\n"
        "Sample point:\n"
        f"  {train_dataset[0]}")

    # Create dataloaders
    train_dataloader = train_dataset.create_dataloader(batch_size=args.batch_size)
    val_dataloader = val_dataset.create_dataloader(batch_size=args.batch_size)
    test_dataloader = test_dataset.create_dataloader(batch_size=args.batch_size)

    batch_X, batch_y = next(iter(train_dataloader))
    
    config.logger.info(
        "Sample batch:\n"
        f"  X: {list(batch_X.size())}\n"
        f"  y: {list(batch_y.size())}")

    # Load embeddings
    embedding_matrix = None

    if args.use_glove:
        if args.embedding_dim not in (50, 100, 200, 300):
            raise Exception(
                "Embedding dim must be in (50, 100, 200, 300) is using GloVe.")
        embeddings_file = os.path.join(
            config.EMBEDDINGS_DIR, f'glove.6B.{args.embedding_dim}d.txt')
        glove_embeddings = utils.load_glove_embeddings(
            embeddings_file=embeddings_file)
        embedding_matrix = utils.make_embeddings_matrix(
            embeddings=glove_embeddings, token_to_index=X_tokenizer.token_to_index,
            embedding_dim=args.embedding_dim)
        config.logger.info(
            "GloVe Embeddings:\n"
            f"{embedding_matrix.shape}")

    # ModelLSTM Artchitecture
    # model = models.ModelLSTM(
    #     embedding_dim=args.embedding_dim, 
    #     vocab_size=vocab_size, 
    #     hidden_dim=args.hidden_dim, 
    #     stacked_layers=args.stacked_layers,
    #     dropout_p=args.dropout_p,
    #     num_classes=len(y_tokenizer.classes), 
    #     pretrained_embeddings=embedding_matrix,
    #     freeze_embeddings=args.freeze_embeddings)

       # // TextCNN Architecture
    model = models.TextCNN(
        embedding_dim=args.embedding_dim, 
        vocab_size=vocab_size,
        num_filters=args.num_filters, 
        filter_sizes=args.filter_sizes,
        hidden_dim=args.hidden_dim, 
        dropout_p=args.dropout_p,
        num_classes=len(y_tokenizer.classes),
        pretrained_embeddings=embedding_matrix,
        freeze_embeddings=args.freeze_embeddings)

    model = model.to(device)
    wandb.watch(model)
    
    config.logger.info(
        "Model:\n"
        f"  {model.named_parameters}")

    # Define optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3)

    # Train
    best_val_loss = train(
        model=model, optimizer=optimizer, scheduler=scheduler,
        num_epochs=args.num_epochs, patience=args.patience,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device)

    # Evaluation
    test_loss, test_acc, y_pred, y_target = test_step(
        model=model, dataloader=test_dataloader, device=device)
    
    config.logger.info(
        "Test performance:\n"
        f"  test_loss: {test_loss:.2f}, test_acc: {test_acc:.1f}")
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc})

    # Per-class performance analysis
    performance = get_performance(y_pred, y_target, classes)

    plot_confusion_matrix(
        y_pred=y_pred, y_target=y_target, classes=classes,
        fp=os.path.join(wandb.run.dir, 'confusion_matrix.png'))

    utils.save_dict(performance, filepath=os.path.join(
        wandb.run.dir, 'performance.json'))
    
    config.logger.info(json.dumps(performance, indent=2, sort_keys=False))

    # Save
    utils.save_dict(args.__dict__, filepath=os.path.join(
        wandb.run.dir, 'config.json'))
    
    X_tokenizer.save(fp=os.path.join(wandb.run.dir, 'X_tokenizer.json'))
    y_tokenizer.save(fp=os.path.join(wandb.run.dir, 'y_tokenizer.json'))