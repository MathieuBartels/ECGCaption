import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
import collections
from ecgnet.network.ecgresnet import ECGResNet
from EcgCaptionGenerator.network.caption_model import Caption_Model
from EcgCaptionGenerator.utils.pycocoevalcap.eval import COCOEvalCap, PTBTokenizer, Cider
from EcgCaptionGenerator.network.utils_model import beam_search

import random
import multiprocessing
import itertools
import sys
class TopDownLSTM(pl.LightningModule):

    def __init__(self, vocab, word_emb_dim, nb_hidden_lstm1, nb_hidden_lstm2, nb_hidden_att,
                 beam_width, max_number_of_tokens,
                 in_length, in_channels, n_grps, N, 
                 num_classes, dropout, first_width, stride, 
                 dilation, loss_weights, learning_rate, image_feature_dim, 
                 **kwargs):
        super().__init__()
        self.vocab_length = len(vocab)
        self.vocab = vocab
        self.pre_train = False
        self.rl = False
        self.beam_width, self.max_number_of_tokens = beam_width, max_number_of_tokens
        self.save_hyperparameters()
        self.model = ECGResNet(in_length, in_channels, 
                               n_grps, N, num_classes, 
                               dropout, first_width, 
                               stride, dilation)
        
        self.model.flatten = Identity()
        self.model.fc1 = Identity()
        self.model.fc2 = Identity()

        self.language_model = Caption_Model(word_emb_dim, nb_hidden_lstm1, nb_hidden_lstm2, nb_hidden_att,
                                            dict_size=len(vocab), image_feature_dim=image_feature_dim, vocab=vocab, tf_ratio=1)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])

    def set_encoder(self, resnet):
        self.pre_train = True
        self.model = resnet

        self.model.flatten = Identity()
        self.model.fc1 = Identity()
        self.model.fc2 = Identity()

    def forward(self, waveforms, targets, lengths):
        _, image_features = self.model(waveforms)
        output = self.language_model(image_features, targets, lengths)

        return output

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
            if self.current_epoch == 0:
                for param in self.model.parameters():
                    param.requires_grad = False

            if self.current_epoch == 5:
                for param in self.model.parameters():
                    param.requires_grad = True
        if self.rl:
            self.Cider = Cider(gts=self.gts)
        
    def training_step(self, batch, batch_idx):
        """Performs a training step.

        Args:
            batch (dict): Output of the dataloader.
            batch_idx (int): Index no. of this batch.

        Returns:
            tensor: Total loss for this step.
        """
        if self.rl:
            return self.train_scst(batch, batch_idx)
        waveforms, _, _, _, targets, lengths = batch
        output = self(waveforms, targets, lengths)
        train_loss = self.loss(output, targets, lengths)
        self.log('train_loss', train_loss.item())
        return train_loss

    def train_scst(self, batch, batch_idx):
        # Training with self-critical
        waveforms, _, _, ids, targets, lengths = batch
        # tokenizer_pool = multiprocessing.Pool()
        nb_batch = waveforms.shape[0]
        max_length = 50
        beam_size = 5
        vocab = self.vocab
        # im features:
        _, image_feats = self.model(waveforms)
        device = image_feats.device

        out, log_probs = beam_search(self.language_model, image_feats, max_length, vocab.word2idx['<end>'], beam_size, out_size=beam_size)

        caps_gt = vocab.decode(targets)

        caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
        caps_gen = vocab.decode(out.view(-1, max_length))
        
        # caps_gen, caps_gt = tokenizer_pool.map(PTBTokenizer.tokenize, [caps_gen, caps_gt])
        caps_gen = PTBTokenizer.tokenize(caps_gen)
        caps_gt = PTBTokenizer.tokenize(caps_gt)
        # print(caps_gen, caps_gt)
        reward = self.Cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
        reward = torch.from_numpy(reward).to(device).view(nb_batch, beam_size)
        reward_baseline = torch.mean(reward, -1, keepdim=True)
        loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

        loss = loss.mean()

        self.log('train_running_loss', loss.item())
        self.log('train_running_reward', reward.mean().item())
        self.log('train_running_reward_baseline', reward_baseline.mean().item())

        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, _, _, _, targets, lengths = batch
        output = self(waveforms, targets, lengths)
        test_loss = self.loss(output, targets, lengths)
        self.log('val_loss', test_loss.item())

        # generated = self.log_inference(batch, batch_idx)

        # true_generated = [self.vocab.idx2word[word.item()] for word in targets[0]]

        # if random.random() < 0.5:
        #     self.logger.experiment.log_text('validation_generation', ' '.join(generated))
        #     self.logger.experiment.log_text('validation_generation_true', ' '.join(true_generated))

        return test_loss

    def log_inference(self, batch, batch_idx):
        waveforms, _, _, ids, targets, lengths = batch
        beamsearch, probs = self.generate(waveforms, self.beam_width, self.max_number_of_tokens, ids)

        return beamsearch, probs

    def on_test_epoch_start(self):
        self.res = {}

    def test_step(self, batch, batch_idx):
        waveforms, _, _, ids, targets, lengths = batch
        max_length = 50
        beam_size = 5
        vocab = self.vocab
        # im features:
        _, image_feats = self.model(waveforms)

        out, _ = beam_search(self.language_model, image_feats, max_length, vocab.word2idx['<end>'], beam_size, out_size=1)
        caps_gen = vocab.decode(out.view(-1, max_length), listfy=True)
        res = dict(zip(ids, caps_gen))

        self.res.update(res)
        
    def on_test_epoch_end(self):
        COCOEval = COCOEvalCap()
        res = collections.OrderedDict(sorted(self.res.items()))
        gts = res
        # gts = collections.OrderedDict(sorted(self.gts.items()))
        # print(res)
        # print(gts)
        COCOEval.evaluate(gts, res)

        evaluators = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"] 
            
        for evaluator in evaluators:
            self.logger.log_metric(f'test_{evaluator}', COCOEval.eval[evaluator])
        return COCOEval.eval

    def generate(self, waveforms, beam_width, max_number_of_tokens, ids):
        _, image_features = self.model(waveforms)
        return self.language_model.beam_search(image_features, max_number_of_tokens, beam_width, ids)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x