import torch
import random
from torch import nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import collections
from ecgnet.network.ecgresnet import ECGResNet
from EcgCaptionGenerator.network.transformer import TransformerModule
from EcgCaptionGenerator.network.utils_model import get_next_word

class Transformer(pl.LightningModule):
    def __init__(self, vocab, in_length, in_channels, 
                n_grps, N, num_classes, 
                dropout, first_width, 
                stride, dilation, num_layers, d_mode, nhead, **kwargs):
        super().__init__()
        self.vocab_length = len(vocab)
        self.vocab = vocab
        self.save_hyperparameters()
        self.model = ECGResNet(in_length, in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation)
        
        self.model.flatten = Identity()
        self.model.fc1 = AveragePool()
        self.model.fc2 = AveragePool()

        self.pre_train = False
        self.feature_embedding = nn.Linear(512, d_mode)
        self.embed = nn.Embedding(len(vocab), d_mode)
        self.transformer = TransformerModule(d_mode, nhead, num_layers)
        self.transformer.apply(init_weights)

        self.to_vocab = nn.Sequential(nn.Linear(d_mode, len(vocab)))

        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])
        self.nlll_criterion = nn.NLLLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])

    def load_pre_trained(self, pre_trained):
        self.model = pre_trained.model
        self.pre_train = True

    def sample(self, waveforms, sample_method, max_length):
        _, (image_features, _) = self.model(waveforms.cuda())

        image_features = image_features.transpose(1, 2).transpose(1, 0)
        image_features = self.feature_embedding(image_features)

        start_tokens = torch.tensor([self.vocab('<start>')], device=image_features.device)
        nb_batch = image_features.shape[1]
        start_tokens = start_tokens.repeat(nb_batch, 1)
        sent = self.embed(start_tokens).transpose(1, 0)
        
        attended_features = None
        
        tgt_mask = torch.zeros(sent.shape[1], sent.shape[0], device=image_features.device, dtype=bool)
        y_out = torch.zeros(nb_batch, max_length, device=image_features.device)
        
        for i in range(max_length):
            out, attended_features = self.transformer.forward_one_step(image_features, sent, tgt_mask, attended_features=attended_features)
            out = self.to_vocab(out[-1, :, :]).squeeze(0)
            s = sample_method
            word_idx, props = get_next_word(out, temp=s['temp'], k=s['k'], p=s['p'], greedy=s['greedy'], m=s['m'])
            y_out[:, i] = word_idx

            ended_mask = (tgt_mask[:, -1] | (word_idx == self.vocab('<end>'))).unsqueeze(1)
            tgt_mask = torch.cat((tgt_mask, ended_mask), dim=1)

            embedded = self.embed(word_idx).unsqueeze(0)
            sent = torch.cat((sent, embedded), dim=0)
            
            if ended_mask.sum() == nb_batch:
                break

        return y_out


    def forward(self, waveforms, targets):
        _, (image_features, _) = self.model(waveforms)
        # print(image_features.shape)
        image_features = image_features.transpose(1, 2).transpose(1,0) #( batch, feature, number)
        image_features = self.feature_embedding(image_features)
        tgt_key_padding_mask = targets == 0

        embedded = self.embed(targets).transpose(1, 0)
        out = self.transformer(image_features, embedded, tgt_key_padding_mask)
        vocab_distribution = self.to_vocab(out)
        return vocab_distribution

    def loss(self, out, targets, run_name):
        if random.random() < 0.1:
            log_text(self, out, targets, run_name, out.shape[1])
        
        # print(out.shape, targets.shape)
        out = F.log_softmax(out, dim=-1).reshape(-1, len(self.vocab))
        target = targets[:, 1:]
        batch_size, seq_length = target.shape
        target = target.transpose(1, 0).reshape(-1)
        
        # print(out.shape, target.shape)
        loss = self.nlll_criterion(out, target)
        loss = loss.reshape(batch_size, seq_length).sum(dim=1).mean(dim=0)
        return loss

    def on_epoch_start(self):
        if self.pre_train:
            for param in self.model.parameters():
                param.requires_grad = self.current_epoch > 0

    def training_step(self, batch, batch_idx):
        waveforms, _, _, _, targets, lengths = batch
        vocab_distribution = self.forward(waveforms, targets)[:-1, :, :]
        run_name = 'training'
        loss = self.loss(vocab_distribution, targets, run_name)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, _, _, _, targets, lengths = batch
        vocab_distribution = self.forward(waveforms, targets)[:-1, :, :]
        run_name = 'validate'
        loss = self.loss(vocab_distribution, targets, run_name)
        self.log('val_loss', loss.item())
        return loss

    def on_test_epoch_start(self):
        self.res = {}

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def log_text(model, words, targets, run_name, nb_batch):
    index = random.randint(0, nb_batch - 1)
    # print(words.shape)
    predicted = torch.max(words, 2)[1]
    predicted = predicted.transpose(1, 0)
    # print(predicted.shape)
    # print(predicted[index, :])
    # print(predicted[index, :].shape)
    # print(model.vocab.idx2word[2])
    model.logger.log_text(f'{run_name}_generations', model.vocab.decode(predicted, skip_first=False)[index])
    model.logger.log_text(f'{run_name}_generations_truths', model.vocab.decode(targets)[index])

class AveragePool(nn.Module):
    def __init__(self, kernel_size=10):
        super(AveragePool, self).__init__()
        
    def forward(self, x):
        signal_size = x.shape[-1]
        kernel = torch.nn.AvgPool1d(signal_size)
        average_feature = kernel(x).squeeze(-1)
        return x, average_feature

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)