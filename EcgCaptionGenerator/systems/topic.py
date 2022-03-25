import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
import collections
from ecgnet.network.ecgresnet import ECGResNet
from EcgCaptionGenerator.network.topic import MLC, CoAttention, WordLSTM, SentenceLSTM
from EcgCaptionGenerator.utils.pycocoevalcap.eval import COCOEvalCap, PTBTokenizer, Cider
from EcgCaptionGenerator.network.utils_model import beam_search

import random
import multiprocessing
import itertools
import sys

class Topic(pl.LightningModule):
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

        self.MLC1 = MLC(classes=100, sementic_features_dim=128, fc_in_features=128, k=15)
        self.attention1 = CoAttention(version='v6', embed_size=128, hidden_size=128, visual_size=128, k=15)
        self.word_lstm1 = WordLSTM(128, 128, len(vocab), 3, n_max=50)
        weight1 = 0.5
        decoder1 = [self.MLC1, self.attention1, self.word_lstm1, weight1]

        self.MLC2 = MLC(classes=100, sementic_features_dim=512, fc_in_features=512, k=15)
        self.attention2 = CoAttention(version='v6', embed_size=512, hidden_size=512, visual_size=512, k=15)
        self.word_lstm2 = WordLSTM(512, 512, len(vocab), 3, n_max=50)
        weight2 = 1 - weight1
        decoder2 = [self.MLC2, self.attention2, self.word_lstm2, weight2]

        self.decoders = [decoder1, decoder2]
        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])
        self.mse_criterion = nn.MSELoss(reduction="none")

    def load_pre_trained(self, pre_trained):
        self.model = pre_trained.model
        self.MLC2 = pre_trained.MLC2
    
        self.decoders[-1] = [self.MLC2, self.attention2, self.word_lstm2, 1]

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
        word_loss = self.ce_criterion(words.permute(0, 2, 1), targets).sum(dim=1)
        self.log(log_name, word_loss.mean().item())
        return word_loss

    def forward_consults(self, waveforms, targets, lengths, run_name='validate'):
        mlc, co_att, word_lstm, weight = self.decoders[-1]
        with torch.no_grad():
            im_features, avg_features = self.model(waveforms)[-1]
            tags, semantic_features = mlc.forward(avg_features)
        
        ctx, alpht_v, alpht_a = co_att.forward(avg_features, semantic_features)

        words = word_lstm.forward(ctx.unsqueeze(1), targets, lengths)[:, :-1, :]
        if random.random() < 0.1:
            nb_batch = ctx.shape[0]
            log_text(self, words, targets, run_name, 0, nb_batch)

        batch_word_loss = self.loss_words(words, targets, f'{run_name}_word_loss') * weight
        
        self.log(f'{run_name}_batch_word_loss', batch_word_loss.mean().item())
        return batch_word_loss.mean()

    def forward(self, waveforms, targets, lengths, label, run_name="validate"):
        image_features = self.model(waveforms)

        batch_tag_loss = 0
        batch_word_loss = 0

        for i, (im_features, avg_features) in enumerate(image_features):
            mlc, co_att, word_lstm, weight = self.decoders[i]

            tags, semantic_features, ctx, alpht_v, alpht_a = self.forward_attention(mlc, co_att, avg_features)

            log_name = f'{run_name}_{i}_tag_loss'
            batch_tag_loss += self.loss_tags(tags, label, log_name) * weight

            words = word_lstm.forward(ctx.unsqueeze(1), targets, lengths)[:, :-1, :]
            if random.random() < 0.1:
                nb_batch = ctx.shape[0]
                log_text(self, words, targets, run_name, i, nb_batch)

            batch_word_loss += self.loss_words(words, targets, f'{run_name}_{i}_word_loss') * weight
        
        # print(batch_tag_loss, batch_word_loss)
        self.log(f'{run_name}_batch_tag_loss', batch_tag_loss.mean().item())
        self.log(f'{run_name}_batch_word_loss', batch_word_loss.mean().item())
        return (batch_tag_loss * 0.3 + batch_word_loss * 0.7).mean()

    def beam(self, waveforms):
        _, (im_features, avg_features) = self.model(waveforms)
        mlc, co_att, word_lstm, weight = self.decoders[-1]

        tags, semantic_features, ctx, alpht_v, alpht_a = self.forward_attention(mlc, co_att, avg_features)
        beam_size = 5
        out, log_probs = beam_search(word_lstm, ctx.unsqueeze(1), 50, self.vocab.word2idx['<end>'], beam_size, out_size=1, start_word=self.vocab.word2idx['<start>'])

        return out, log_probs

    def sample(self, waveforms, s):
        with torch.no_grad():
            _, (im_features, avg_features) = self.model(waveforms)
            mlc, co_att, word_lstm, weight = self.decoders[-1]

            tags, semantic_features, ctx, alpht_v, alpht_a = self.forward_attention(mlc, co_att, avg_features)

            start_word = torch.tensor([self.vocab('<start>')], device=waveforms.device)
            start_word = start_word.repeat(tags.shape[0], 1)
            words = word_lstm.sample(ctx.unsqueeze(1), start_word, s=s)[:, :]
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
                param.requires_grad = False

            for param in self.MLC2.parameters():
                param.requires_grad = False

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
        # waveforms, _, _, _, targets, lengths, labels = batch
        # max_length = 50
        # beam_size = 5
        # vocab = self.vocab
        # # im features:
        # _, image_feats = self.model(waveforms)

        # out, _ = beam_search(self.language_model, image_feats, max_length, vocab.word2idx['<end>'], beam_size, out_size=1)
        # caps_gen = vocab.decode(out.view(-1, max_length), listfy=True)
        # res = dict(zip(ids, caps_gen))

        # self.res.update(res)
        
    def on_test_epoch_end(self):
        pass
        # COCOEval = COCOEvalCap()
        # res = collections.OrderedDict(sorted(self.res.items()))
        # gts = collections.OrderedDict(sorted(self.gts.items()))
        # # print(res)
        # # print(gts)
        # COCOEval.evaluate(gts, res)

        # evaluators = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"] 
            
        # for evaluator in evaluators:
        #     self.logger.log_metric(f'test_{evaluator}', COCOEval.eval[evaluator])
        # return COCOEval.eval

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def log_text(model, words, targets, run_name, i, nb_batch):
    index = random.randint(0, nb_batch - 1)
    predicted = torch.max(words, 2)[1]
    model.logger.log_text(f'{run_name}_{i}_generations', model.vocab.decode(predicted)[index])
    model.logger.log_text(f'{run_name}_{i}_generations_truths', model.vocab.decode(targets)[index])

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