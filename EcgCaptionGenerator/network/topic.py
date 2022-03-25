import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
from EcgCaptionGenerator.network.utils_model import get_next_word

class MLC(nn.Module):
    def __init__(self,
                 classes=156,
                 sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax(dim=-1)
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features


class CoAttention(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,
                 hidden_size=512,
                 visual_size=2048,
                 k=10,
                 momentum=0.1):
        super(CoAttention, self).__init__()
        self.version = version
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=k, momentum=momentum)

        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

        self.W_a.weight.data.uniform_(-0.1, 0.1)
        self.W_a.bias.data.fill_(0)

        self.W_a_h.weight.data.uniform_(-0.1, 0.1)
        self.W_a_h.bias.data.fill_(0)

        self.W_a_att.weight.data.uniform_(-0.1, 0.1)
        self.W_a_att.bias.data.fill_(0)

        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def forward(self, avg_features, semantic_features, h_sent=None):
        if self.version == 'v1':
            return self.v1(avg_features, semantic_features, h_sent)
        elif self.version == 'v2':
            return self.v2(avg_features, semantic_features, h_sent)
        elif self.version == 'v3':
            return self.v3(avg_features, semantic_features, h_sent)
        elif self.version == 'v4':
            return self.v4(avg_features, semantic_features, h_sent)
        elif self.version == 'v5':
            return self.v5(avg_features, semantic_features, h_sent)
        elif self.version == 'v6':
            return self.v6(avg_features, semantic_features)

    def v6(self, avg_features, semantic_features) -> object:
        """
        only training
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v1(self, avg_features, semantic_features, h_sent) -> object:
        """
        only training
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v2(self, avg_features, semantic_features, h_sent) -> object:
        """
        no bn
        :rtype: object
        """
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v3(self, avg_features, semantic_features, h_sent) -> object:
        """
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v4(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(torch.add(W_v, W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v5(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(self.bn_v(torch.add(W_v, W_v_h)))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(self.bn_a(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a


class SentenceLSTM(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,
                 hidden_size=512,
                 num_layers=1,
                 dropout=0.3,
                 momentum=0.1):
        super(SentenceLSTM, self).__init__()
        self.version = version

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)

        self.W_t_h = nn.Linear(in_features=hidden_size,
                               out_features=embed_size,
                               bias=True)
        self.bn_t_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_t_ctx = nn.Linear(in_features=embed_size,
                                 out_features=embed_size,
                                 bias=True)
        self.bn_t_ctx = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s_1 = nn.Linear(in_features=hidden_size,
                                    out_features=embed_size,
                                    bias=True)
        self.bn_stop_s_1 = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s = nn.Linear(in_features=hidden_size,
                                  out_features=embed_size,
                                  bias=True)
        self.bn_stop_s = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop = nn.Linear(in_features=embed_size,
                                out_features=2,
                                bias=True)
        self.bn_stop = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_topic = nn.Linear(in_features=embed_size,
                                 out_features=embed_size,
                                 bias=True)
        self.bn_topic = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.__init_weight()

    def __init_weight(self):
        self.W_t_h.weight.data.uniform_(-0.1, 0.1)
        self.W_t_h.bias.data.fill_(0)

        self.W_t_ctx.weight.data.uniform_(-0.1, 0.1)
        self.W_t_ctx.bias.data.fill_(0)

        self.W_stop_s_1.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s_1.bias.data.fill_(0)

        self.W_stop_s.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s.bias.data.fill_(0)

        self.W_stop.weight.data.uniform_(-0.1, 0.1)
        self.W_stop.bias.data.fill_(0)

        self.W_topic.weight.data.uniform_(-0.1, 0.1)
        self.W_topic.bias.data.fill_(0)

    def forward(self, ctx, prev_hidden_state, states=None) -> object:
        """
        :rtype: object
        """
        if self.version == 'v1':
            return self.v1(ctx, prev_hidden_state, states)
        elif self.version == 'v2':
            return self.v2(ctx, prev_hidden_state, states)
        elif self.version == 'v3':
            return self.v3(ctx, prev_hidden_state, states)

    def v1(self, ctx, prev_hidden_state, states=None):
        """
        v1 (only training)
        :param ctx:
        :param prev_hidden_state:
        :param states:
        :return:
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.W_topic(self.sigmoid(self.bn_t_h(self.W_t_h(hidden_state))
                                          + self.bn_t_ctx(self.W_t_ctx(ctx))))
        p_stop = self.W_stop(self.sigmoid(self.bn_stop_s_1(self.W_stop_s_1(prev_hidden_state))
                                          + self.bn_stop_s(self.W_stop_s(hidden_state))))
        return topic, p_stop, hidden_state, states

    def v2(self, ctx, prev_hidden_state, states=None):
        """
        v2
        :rtype: object
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.bn_topic(self.W_topic(self.tanh(self.bn_t_h(self.W_t_h(hidden_state)
                                                                 + self.W_t_ctx(ctx)))))
        p_stop = self.bn_stop(self.W_stop(self.tanh(self.bn_stop_s(self.W_stop_s_1(prev_hidden_state)
                                                                   + self.W_stop_s(hidden_state)))))
        return topic, p_stop, hidden_state, states

    def v3(self, ctx, prev_hidden_state, states=None):
        """
        v3
        :rtype: object
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.W_topic(self.tanh(self.W_t_h(hidden_state) + self.W_t_ctx(ctx)))
        p_stop = self.W_stop(self.tanh(self.W_stop_s_1(prev_hidden_state) + self.W_stop_s(hidden_state)))
        return topic, p_stop, hidden_state, states


class WordLSTM(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 n_max=50):
        super(WordLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()
        self.n_max = n_max
        self.vocab_size = vocab_size

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, topic_vec, captions, lengths):
        # print(captions.shape)
        embeddings = self.embed(captions)
        # print(embeddings.shape)
        embeddings = torch.cat((topic_vec, embeddings), 1)
        lengths = np.array(lengths) + 1
        packed_targets = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        # print(embeddings.shape)
        packed_output, _ = self.lstm(packed_targets)

        hidden, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # print(hidden.shape)
        outputs = self.linear(hidden[:, :, :])
        # print(outputs.shape)
        return outputs

    def init_inference(self, nb_batch, cuda):
        start_word = torch.tensor([self.vocab('<start>')])
        start_word = start_word.cuda() if cuda else start_word
        start_word = self.embedding(start_word)

        start_word = start_word.repeat(nb_batch, 1)

        if cuda:
            start_word = start_word.cuda()
        return start_word

    def forward_one_step(self, state, prev_output):
        try:
            embeddings = self.embed(prev_output)
        except:
            print(prev_output)
        print('embeddings', embeddings.shape)
        # print(embeddings.shape)
        hidden, state = self.lstm(embeddings, state)
        print('hidden', hidden.shape)
        # print(hidden.shape)
        outputs = self.linear(hidden[:, :, :])
        print('outputs', outputs.shape)
        # print(outputs.shape)
        state = [s.permute(1,0,2) for s in state]
        return outputs, state

    def step(self, t, prev_output, state, context, seq, mode='teacher_forcing', **kwargs):
        
        if t == 0:
            hidden, state = self.lstm(context)
        else:
            state = [s.permute(1,0,2).contiguous() for s in state]

        return self.forward_one_step(state, prev_output)

    def sample(self, features, start_tokens, s=None):
        sampled_ids = np.zeros((np.shape(features)[0], self.n_max))
        sampled_ids[:, 0] = start_tokens.view(-1, ).cpu()
        predicted = start_tokens
        _, hidden = self.lstm(features)
        for i in range(1, self.n_max):
            embeddings = self.embed(predicted)
            hidden_states, hidden = self.lstm(embeddings, hidden)
            # print(hidden_states.shape)
            hidden_states = hidden_states[:, -1, :]
            # print(hidden_states.shape)
            outputs = self.linear(hidden_states)
            # print(outputs.shape)
            if s:
                predicted, next_logprobs = get_next_word(outputs, s['temp'], s['k'], s['p'], s['greedy'], s['m'])
                # print(predicted)
            else:
                predicted, next_logprobs = get_next_word(outputs)

            # print(predicted.shape)
            predicted = predicted
            sampled_ids[:, i] = predicted.cpu()
            predicted = predicted.unsqueeze(1)
        return sampled_ids

class SentenceLSTM(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,
                 hidden_size=512,
                 num_layers=1,
                 dropout=0.3,
                 momentum=0.1):
        super(SentenceLSTM, self).__init__()
        self.version = version

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)

        self.W_t_h = nn.Linear(in_features=hidden_size,
                               out_features=embed_size,
                               bias=True)
        self.bn_t_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_t_ctx = nn.Linear(in_features=embed_size,
                                 out_features=embed_size,
                                 bias=True)
        self.bn_t_ctx = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s_1 = nn.Linear(in_features=hidden_size,
                                    out_features=embed_size,
                                    bias=True)
        self.bn_stop_s_1 = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s = nn.Linear(in_features=hidden_size,
                                  out_features=embed_size,
                                  bias=True)
        self.bn_stop_s = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop = nn.Linear(in_features=embed_size,
                                out_features=2,
                                bias=True)
        self.bn_stop = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_topic = nn.Linear(in_features=embed_size,
                                 out_features=embed_size,
                                 bias=True)
        self.bn_topic = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.__init_weight()

    def __init_weight(self):
        self.W_t_h.weight.data.uniform_(-0.1, 0.1)
        self.W_t_h.bias.data.fill_(0)

        self.W_t_ctx.weight.data.uniform_(-0.1, 0.1)
        self.W_t_ctx.bias.data.fill_(0)

        self.W_stop_s_1.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s_1.bias.data.fill_(0)

        self.W_stop_s.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s.bias.data.fill_(0)

        self.W_stop.weight.data.uniform_(-0.1, 0.1)
        self.W_stop.bias.data.fill_(0)

        self.W_topic.weight.data.uniform_(-0.1, 0.1)
        self.W_topic.bias.data.fill_(0)

    def forward(self, ctx, prev_hidden_state, states=None) -> object:
        """
        :rtype: object
        """
        if self.version == 'v1':
            return self.v1(ctx, prev_hidden_state, states)
        elif self.version == 'v2':
            return self.v2(ctx, prev_hidden_state, states)
        elif self.version == 'v3':
            return self.v3(ctx, prev_hidden_state, states)

    def v1(self, ctx, prev_hidden_state, states=None):
        """
        v1 (only training)
        :param ctx:
        :param prev_hidden_state:
        :param states:
        :return:
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.W_topic(self.sigmoid(self.bn_t_h(self.W_t_h(hidden_state))
                                          + self.bn_t_ctx(self.W_t_ctx(ctx))))
        p_stop = self.W_stop(self.sigmoid(self.bn_stop_s_1(self.W_stop_s_1(prev_hidden_state))
                                          + self.bn_stop_s(self.W_stop_s(hidden_state))))
        return topic, p_stop, hidden_state, states

    def v2(self, ctx, prev_hidden_state, states=None):
        """
        v2
        :rtype: object
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.bn_topic(self.W_topic(self.tanh(self.bn_t_h(self.W_t_h(hidden_state)
                                                                 + self.W_t_ctx(ctx)))))
        p_stop = self.bn_stop(self.W_stop(self.tanh(self.bn_stop_s(self.W_stop_s_1(prev_hidden_state)
                                                                   + self.W_stop_s(hidden_state)))))
        return topic, p_stop, hidden_state, states

    def v3(self, ctx, prev_hidden_state, states=None):
        """
        v3
        :rtype: object
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.W_topic(self.tanh(self.W_t_h(hidden_state) + self.W_t_ctx(ctx)))
        p_stop = self.W_stop(self.tanh(self.W_stop_s_1(prev_hidden_state) + self.W_stop_s(hidden_state)))
        return topic, p_stop, hidden_state, states