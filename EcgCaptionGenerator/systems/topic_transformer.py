import torch
import random
from torch import nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import collections
from ecgnet.network.ecgresnet import ECGResNet
from EcgCaptionGenerator.network.topic import MLC, CoAttention, WordLSTM, SentenceLSTM
from EcgCaptionGenerator.network.transformer_topic import TopicTransformerModule
from EcgCaptionGenerator.network.utils_model import get_next_word

class TopicTransformer(pl.LightningModule):
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
        self.embed = nn.Embedding(len(vocab), 2*d_mode)

        mlc = MLC(classes=100, sementic_features_dim=d_mode, fc_in_features=512, k=15)
        attention = CoAttention(version='v6', embed_size=d_mode, hidden_size=d_mode, visual_size=512, k=15)

        self.transformer = TopicTransformerModule(d_mode, nhead, num_layers, mlc, attention)
        self.transformer.apply(init_weights)

        self.to_vocab = nn.Sequential(nn.Linear(2*d_mode, len(vocab)))

        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])
        self.nlll_criterion = nn.NLLLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])
        self.mse_criterion = nn.MSELoss(reduction="none")

    def load_pre_trained(self, pre_trained):
        self.transformer.mlc = pre_trained.transformer.mlc
        self.transformer.attention = pre_trained.transformer.attention
        self.transformer.transformer_encoder = pre_trained.transformer.transformer_encoder
        self.model = pre_trained.model

        self.pre_train = True

        _, (image_features, avg_feats) = self.model(waveforms)

        tgt_key_padding_mask = targets == 0

        embedded = self.embed(targets).transpose(1, 0)
        out, tags = self.transformer(image_features, avg_feats, embedded, tgt_key_padding_mask)

        vocab_distribution = self.to_vocab(out)


    def sample(self, waveforms, sample_method, max_length):
        _, (image_features, avg_feats) = self.model(waveforms.cuda())

        image_features = image_features.transpose(1, 2).transpose(1, 0)
        image_features = self.feature_embedding(image_features)

        start_tokens = torch.tensor([self.vocab('<start>')], device=image_features.device)
        nb_batch = waveforms.shape[0]
        start_tokens = start_tokens.repeat(nb_batch, 1)
        sent = self.embed(start_tokens).transpose(1, 0)
        
        attended_features = None
        
        tgt_mask = torch.zeros(sent.shape[1], sent.shape[0], device=image_features.device, dtype=bool)
        y_out = torch.zeros(nb_batch, max_length, device=image_features.device)
        
        for i in range(max_length):
            out, attended_features = self.transformer.forward_one_step(image_features, avg_feats, sent, tgt_mask, attended_features=attended_features)
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
        _, (image_features, avg_feats) = self.model(waveforms)
        # print(image_features.shape)
        image_features = image_features.transpose(1, 2).transpose(1,0) #( batch, feature, number)
        image_features = self.feature_embedding(image_features)
        tgt_key_padding_mask = targets == 0

        embedded = self.embed(targets).transpose(1, 0)
        out, tags = self.transformer(image_features, avg_feats, embedded, tgt_key_padding_mask)

        vocab_distribution = self.to_vocab(out)
        return vocab_distribution, tags

    def loss_tags(self, tags, label):
        label.requires_grad = False
        tag_loss = self.mse_criterion(tags, label).sum(dim=1)
        return tag_loss

    def loss(self, out, targets, tags, topic, run_name):
        if random.random() < 0.1:
            log_text(self, out, targets, run_name, out.shape[1])
        
        # print(out.shape, targets.shape)
        out = F.log_softmax(out, dim=-1).reshape(-1, len(self.vocab))
        target = targets[:, 1:]
        batch_size, seq_length = target.shape
        target = target.transpose(1, 0).reshape(-1)
        
        # print(out.shape, target.shape)
        loss = self.nlll_criterion(out, target)
        loss = loss.reshape(batch_size, seq_length).sum(dim=1)

        if self.pre_train:
            return loss.mean(dim=0)

        target_loss = self.loss_tags(tags, topic)

        self.log(f'{run_name}_batch_tag_loss', target_loss.mean().item())
        self.log(f'{run_name}_batch_word_loss', loss.mean().item())
        return (target_loss * 0.3 + loss * 0.7).mean()

    def on_epoch_start(self):
        if self.pre_train:
            for param in self.transformer.mlc.parameters():
                param.requires_grad = False
            for param in self.transformer.attention.parameters():
                param.requires_grad = False
            for param in self.transformer.transformer_encoder.parameters():
                param.requires_grad = self.current_epoch > 5
            for param in self.model.parameters():
                param.requires_grad = self.current_epoch > 5

    def training_step(self, batch, batch_idx):
        waveforms, _, _, _, targets, lengths, labels = batch
        vocab_distribution, tags = self.forward(waveforms, targets)
        vocab_distribution = vocab_distribution[:-1, :, :]
        run_name = 'training'
        loss = self.loss(vocab_distribution, targets, tags, labels, run_name)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, _, _, _, targets, lengths, labels = batch
        vocab_distribution, tags = self.forward(waveforms, targets)
        vocab_distribution = vocab_distribution[:-1, :, :]
        run_name = 'validate'
        loss = self.loss(vocab_distribution, targets, tags, labels, run_name)
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