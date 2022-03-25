import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
import collections
from ecgnet.network.ecgresnet import ECGResNet
from EcgCaptionGenerator.network.caption_model import Caption_Model
from EcgCaptionGenerator.network.topic import MLC, CoAttention, WordLSTM, SentenceLSTM
from EcgCaptionGenerator.utils.pycocoevalcap.eval import COCOEvalCap, PTBTokenizer, Cider
from EcgCaptionGenerator.network.utils_model import beam_search

import random
import multiprocessing
import itertools
import sys

class TopicSimDecoder(pl.LightningModule):
    def __init__(self, vocab, word_emb_dim, nb_hidden_lstm1, nb_hidden_lstm2, nb_hidden_att,
                 beam_width, max_number_of_tokens,
                 in_length, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, loss_weights, learning_rate, image_feature_dim, 
                 **kwargs):
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

        self.MLC = MLC(classes=100, sementic_features_dim=512, fc_in_features=512, k=15)
        self.attention = CoAttention(version='v6', embed_size=512, hidden_size=512, visual_size=512, k=15)
        self.language_model = Caption_Model(word_emb_dim, nb_hidden_lstm1, nb_hidden_lstm2, nb_hidden_att,
                                            dict_size=len(vocab), image_feature_dim=image_feature_dim, vocab=vocab, tf_ratio=1)

        self.decoder = [self.MLC, self.attention, self.language_model]
        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])
        self.mse_criterion = nn.MSELoss(reduction="none")

    def load_pre_trained(self, pre_trained):
        self.model = pre_trained.model
        self.MLC = pre_trained.MLC
    
        self.decoder = [self.MLC, self.attention, self.language_model]

    def forward_attention(self, mlc, co_att, avg_features):
        tags, semantic_features = mlc.forward(avg_features)
        ctx, alpht_v, alpht_a = co_att.forward(avg_features, semantic_features)
        return tags, semantic_features, ctx, alpht_v, alpht_a

    def loss_tags(self, tags, label, log_name):
        label.requires_grad = False
        tag_loss = self.mse_criterion(tags, label).sum(dim=1)
        self.log(log_name, tag_loss.mean().item())
        return tag_loss

    def loss_words(self, words, targets, log_name):
        word_loss = self.ce_criterion(words[:, :-1, :].permute(0, 2, 1), targets[:, 1:]).sum(dim=1)
        self.log(log_name, word_loss.mean().item())
        return word_loss

    def forward(self, waveforms, targets, lengths, label, run_name="validate"):
        _, (im_features, avg_features) = self.model(waveforms)
        im_features = im_features.transpose(2,1)
        mlc, co_att, language_model = self.decoder

        tags, semantic_features, ctx, alpht_v, alpht_a = self.forward_attention(mlc, co_att, avg_features)

        log_name = f'{run_name}_tag_loss'
        batch_tag_loss = self.loss_tags(tags, label, log_name)

        words = language_model(im_features, targets, lengths, ctx=ctx)
        if random.random() < 0.1:
            nb_batch = ctx.shape[0]
            log_text(self, words, targets, run_name, nb_batch)

        batch_word_loss = self.loss_words(words, targets, f'{run_name}_word_loss')
        
        # print(batch_tag_loss, batch_word_loss)
        self.log(f'{run_name}_batch_tag_loss', batch_tag_loss.mean().item())
        self.log(f'{run_name}_batch_word_loss', batch_word_loss.mean().item())
        return (batch_tag_loss * 0.3 + batch_word_loss * 0.7).mean()


    def sample(self, waveforms, ids, s):
        with torch.no_grad():
            _, (im_features, avg_features) = self.model(waveforms)
            im_features = im_features.transpose(2,1)
            mlc, co_att, language_model = self.decoder

            tags, semantic_features, ctx, alpht_v, alpht_a = self.forward_attention(mlc, co_att, avg_features)

            start_word = torch.tensor([self.vocab('<start>')], device=waveforms.device)
            start_word = start_word.repeat(tags.shape[0], 1)
            words = language_model.generate(im_features, 50, 5, ids, s, ctx=ctx)
            return tags, words

    def check_prediction(self, out, targets):
        nb_batch = out.shape[0]
        index = random.randint(0, nb_batch-1)
        
        val, idx = out[index].max(dim=1)
        pred = ' '.join([self.vocab.idx2word[idxn.item()] for idxn in idx])
        truth = ' '.join([self.vocab.idx2word[word.item()] for word in targets[index]])
        
        self.logger.log_text('train_pred', pred)
        self.logger.log_text('train_truth', truth)

    def loss(self, out, targets, lengths):
        n_ex, vocab_len = out.view(-1, self.vocab_length).shape
        captions = targets
        if random.random() < 0.05:
            self.check_prediction(out, targets)
        return self.loss_fn(out.permute(0, 2, 1), captions).sum(dim=1).mean(dim=0)

    def on_epoch_start(self):
        if self.pre_train:
            for param in self.model.parameters():
                param.requires_grad = self.current_epoch > 30

            for param in self.MLC.parameters():
                param.requires_grad = self.current_epoch > 30

    def training_step(self, batch, batch_idx):
        """Performs a training step.

        Args:
            batch (dict): Output of the dataloader.
            batch_idx (int): Index no. of this batch.

        Returns:
            tensor: Total loss for this step.
        """
        if self.pre_train:
            waveforms, _, _, _, targets, lengths, _ = batch
            loss = self.forward_consults(waveforms, targets, lengths, run_name='train')
            self.log('train_loss', loss.item())
            return loss
        waveforms, _, _, _, targets, lengths, labels = batch
        loss = self(waveforms, targets, lengths, labels, run_name='train')
        # print(loss.dtype)
        # train_loss = self.loss(output, targets, lengths)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        if self.pre_train:
            waveforms, _, _, _, targets, lengths, _ = batch
            loss = self.forward_consults(waveforms, targets, lengths, run_name='train')
            self.log('val_loss', loss.item())
            return loss
        waveforms, _, _, _, targets, lengths, labels = batch
        loss = self(waveforms, targets, lengths, labels)
        # test_loss = self.loss(output, targets, lengths)
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

    
    def forward_consults(self, waveforms, targets, lengths, run_name='validate'):
        with torch.no_grad():
            _, (im_features, avg_features) = self.model(waveforms)
            im_features = im_features.transpose(2,1)
            
            mlc, co_att, language_model = self.decoder
            tags, semantic_features = mlc.forward(avg_features)
        
        ctx, alpht_v, alpht_a = co_att.forward(avg_features, semantic_features)

        words = language_model(im_features, targets, lengths, ctx=ctx)
        if random.random() < 0.1:
            nb_batch = ctx.shape[0]
            log_text(self, words, targets, run_name, nb_batch)

        batch_word_loss = self.loss_words(words, targets, f'{run_name}_word_loss')
        
        # print(batch_tag_loss, batch_word_loss)
        return batch_word_loss.mean()

def log_text(model, words, targets, run_name, nb_batch):
    index = random.randint(0, nb_batch - 1)
    predicted = torch.max(words, 2)[1]
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