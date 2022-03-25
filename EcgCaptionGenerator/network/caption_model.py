import numpy as np
import random
import torch
import torch.nn as nn


from EcgCaptionGenerator.network.utils_model import make_zeros, get_next_word
from EcgCaptionGenerator.network.container import Module

#########################################
#               MAIN MODEL              #
#########################################
class Caption_Model(Module):
    def __init__(self, word_emb_dim, nb_hidden_lstm1, nb_hidden_lstm2, nb_hidden_att, 
                 dict_size, image_feature_dim, vocab, tf_ratio):
        super(Caption_Model, self).__init__()
        self.dict_size = dict_size
        self.image_feature_dim = image_feature_dim
        self.vocab = vocab
        self.tf_ratio = tf_ratio

        self.embed_word = nn.Linear(dict_size, word_emb_dim, bias=False)
        self.embedding = nn.Embedding(dict_size, word_emb_dim)
        self.lstm1 = Attention_LSTM(word_emb_dim,
                                    nb_hidden_lstm2,
                                    image_feature_dim,
                                    nb_hidden_lstm1)
        self.lstm2 = Language_LSTM(nb_hidden_lstm1,
                                   image_feature_dim,
                                   nb_hidden_lstm2)
        self.attention = Visual_Attention(image_feature_dim,
                                          nb_hidden_lstm1,
                                          nb_hidden_att)
        self.predict_word = Predict_Word(nb_hidden_lstm2, dict_size)

        self.h1 = torch.nn.Parameter(torch.zeros(1, nb_hidden_lstm1))
        self.c1 = torch.nn.Parameter(torch.zeros(1, nb_hidden_lstm1))
        self.h2 = torch.nn.Parameter(torch.zeros(1, nb_hidden_lstm2))
        self.c2 = torch.nn.Parameter(torch.zeros(1, nb_hidden_lstm2))
    
    def forward(self, image_feats, true_words, lengths, ctx=None, beam=None):

        nb_timesteps = lengths[0]

        # %TODO A choice was made here on the dimensions of the visual features
        nb_batch, nb_image_feats, _ = image_feats.size()
        
        v_mean = ctx
        if ctx is None:
            v_mean = image_feats.mean(dim=1)
            
        # print("vmean", v_mean.shape)
        use_cuda = image_feats.is_cuda

        # print(f"shape true_words {true_words.shape}")
        embedded_all = self.embedding(true_words)
        # print(f"shape embedded all {embedded_all.shape}")

        packed_targets = torch.nn.utils.rnn.pack_padded_sequence(embedded_all, lengths, batch_first=True)
        batch, batch_size = packed_targets[0], packed_targets[1]
        # print("packed targets and shape: ", packed_targets)

        state, _ = self.init_inference(nb_batch, use_cuda)
        y_out = make_zeros((nb_batch, nb_timesteps, self.dict_size), cuda=use_cuda)

        for t in range(nb_timesteps):
            cur = batch_size[:t].sum()
            size = batch_size[t]
            current_word = batch[cur: cur + size]
            # print(f"timestep is {t}")

            # print(f"vmean shape {v_mean.shape}")
            # print(f"image feats shape {image_feats.shape}")
            v_mean = v_mean[:size, :]
            image_feats = image_feats[:size, : , :]
            # print(f"state shapes {[s.shape for s in state]}")
            state = [s[:size, :] for s in state]
            # print(f"states shapes becomes {[s.shape for s in state]}")
            y, state = self.forward_one_step(state,
                                             current_word,
                                             v_mean,
                                             image_feats)

            # print(f"y shape is {y.shape}")
            y = y[:size, :]
            # print(f"y shape for current timestep becomes: {y.shape} because size is {size}")

            # print(f"yout shape is {y_out.shape}")
            y_out[:size, t, :] = y

            # print(f"targets where shape {batch.shape}")
            targets = batch[cur: cur + size]
            # print(f"but becomes {targets.shape}")
            current_word = self.update_current_word(y, targets)

        return y_out

    def forward_one_step(self, state, word_emb, v_mean, image_feats):
        h1, c1, h2, c2 = state
        h1, c1 = self.lstm1(h1, c1, h2, v_mean, word_emb)
        v_hat = self.attention(image_feats, h1)
        h2, c2 = self.lstm2(h2, c2, h1, v_hat)
        y = self.predict_word(h2)
        state = [h1, c1, h2, c2]
        return y, state

    def update_current_word(self, y, targets):
        use_tf = True if random.random() < self.tf_ratio else False
        if use_tf:
            current_word = targets
        else:
            current_word = torch.argmax(y, dim=1)
            current_word = self.embedding(current_word)
        return current_word

    def generate(self, image_features, max_nb_words, beam_width, ids, s, ctx=None):
        ids = np.array(ids)
        res = {}
        # im features:
        nb_batch = image_features.shape[0]
        v_mean = ctx
        if ctx is None:
            v_mean = image_features.mean(dim=1)

        # init language model :
        use_cuda = image_features.is_cuda
        nb_batch, nb_image_features, _ = image_features.size()
        state, start_word = self.init_inference(nb_batch, use_cuda)

        result = torch.zeros(nb_batch, max_nb_words, device=image_features.device).long()
        current_word = start_word
        all_probs = []
        selection = torch.ones(v_mean.shape[0]) == 1
        lengths = - np.ones(v_mean.shape[0], dtype=int)
        for i in range(max_nb_words):
            logits, state = self.forward_one_step(state, current_word, v_mean, image_features)

            word_idx, props = get_next_word(logits, temp=s['temp'], k=s['k'], p=s['p'], greedy=s['greedy'], m=s['m'])
            
            all_probs.append(props)

            result[selection, i] = word_idx[selection]
            
            for j, idx in enumerate(word_idx):
                if idx == self.vocab.word2idx['<end>']:
                    selection[j] = False
                    if lengths[j] == -1:
                        lengths[j] = i
            current_word = self.embedding(word_idx)
            if selection.sum() == 0:
                break


        all_probs = torch.cat(all_probs, dim=1)
        sent_probs = []
        for i in range(result.shape[0]):
            res[ids[i]] = [' '.join([self.vocab.idx2word[idx.item()] for idx in result[i, :lengths[i]]])]
            sent_probs.append(all_probs[i, 1:lengths[i]].mean().unsqueeze(0))

        return res, torch.cat(sent_probs)

    def step(self, t, prev_output, state, visual, seq, mode='teacher_forcing', **kwargs):
        word_emb = None

        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.image_feats = visual
                self.v_mean = visual.mean(dim=1)
                if isinstance(visual, torch.Tensor):
                    state, word_emb = self.init_inference(visual.shape[0], visual.is_cuda)
            else:
                word_emb = self.embedding(prev_output).squeeze(1)

        return self.forward_one_step(state, word_emb, visual.mean(dim=1), visual)

    def init_inference(self, nb_batch, cuda):
        start_word = torch.tensor([self.vocab('<start>')])
        start_word = start_word.cuda() if cuda else start_word
        start_word = self.embedding(start_word)

        start_word = start_word.repeat(nb_batch, 1)

        if cuda:
            start_word = start_word.cuda()

        h1 = self.h1.repeat(nb_batch, 1)
        c1 = self.c1.repeat(nb_batch, 1)
        h2 = self.h2.repeat(nb_batch, 1)
        c2 = self.c2.repeat(nb_batch, 1)
        state = [h1, c1, h2, c2]

        return state, start_word

#####################################################
#               LANGUAGE SUB-MODULES                #
#####################################################
class Attention_LSTM(nn.Module):
    def __init__(self, dim_word_emb, dim_lang_lstm, dim_image_feats, nb_hidden):
        super(Attention_LSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(dim_lang_lstm + dim_image_feats + dim_word_emb,
                                     nb_hidden,
                                     bias=True)
        
    def forward(self, h1, c1, h2, v_mean, word_emb):
        # print('h2', h2.shape)
        # print('v_mean', v_mean.shape)
        # print('word_emb', word_emb.shape)

        input_feats = torch.cat((h2, v_mean, word_emb),dim=1)
        h_out, c_out = self.lstm_cell(input_feats, (h1, c1))

        return h_out, c_out

class Language_LSTM(nn.Module):
    def __init__(self, dim_att_lstm, dim_visual_att, nb_hidden):
        super(Language_LSTM,self).__init__()
        self.lstm_cell = nn.LSTMCell(dim_att_lstm + dim_visual_att,
                                     nb_hidden,
                                     bias=True)
        
    def forward(self, h2, c2, h1, v_hat):
        input_feats = torch.cat((h1, v_hat),dim=1)
        h_out, c_out = self.lstm_cell(input_feats, (h2, c2))
        return h_out, c_out


class Visual_Attention(nn.Module):
    def __init__(self, dim_image_feats, dim_att_lstm, nb_hidden):
        super(Visual_Attention,self).__init__()
        self.fc_image_feats = nn.Linear(dim_image_feats, nb_hidden, bias=False)
        self.fc_att_lstm = nn.Linear(dim_att_lstm, nb_hidden, bias=False)
        self.act_tan = nn.Tanh()
        self.fc_att = nn.Linear(nb_hidden, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_feats, h1):
        nb_batch, nb_feats, feat_dim = image_feats.size()
        att_lstm_emb = self.fc_att_lstm(h1).unsqueeze(1)
        image_feats_emb = self.fc_image_feats(image_feats)
        all_feats_emb = image_feats_emb + att_lstm_emb.repeat(1,nb_feats,1)

        activate_feats = self.act_tan(all_feats_emb)
        unnorm_attention = self.fc_att(activate_feats)
        normed_attention = self.softmax(unnorm_attention)

        weighted_feats = normed_attention * image_feats

        attended_image_feats = weighted_feats.sum(dim=1)

        return attended_image_feats

class Predict_Word(nn.Module):
    def __init__(self, dim_language_lstm, dict_size):
        super(Predict_Word, self).__init__()
        self.fc = nn.Linear(dim_language_lstm, dict_size)
        
    def forward(self, h2):
        y = self.fc(h2)
        return y

def test_for_nan(x, name="No name given"):
    if torch.isnan(x).sum() > 0:
        print("{} has NAN".format(name))
        exit()
