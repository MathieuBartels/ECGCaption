import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.gumbel import Gumbel
import numpy as np

from EcgCaptionGenerator.network.typing import *
from EcgCaptionGenerator.network.beamsearch import BeamSearch

def make_zeros(shape, cuda=False):
    zeros = torch.zeros(shape)
    if cuda:
        zeros = zeros.cuda()
    return zeros

def get_next_word(logits, temp=None, k=None, p=None, greedy=None, m=None):
    probs = F.softmax(logits, dim=-1)
    logprobs = F.log_softmax(logits, dim=-1)

    if temp is not None:
        samp_probs = F.softmax(logits.div_(temp), dim=-1)
    else:
        samp_probs = probs.clone()

    if greedy:
        next_probs, next_tokens = probs.topk(1)
        next_logprobs = logprobs.gather(1, next_tokens.view(-1, 1))

    elif k is not None:
        indices_to_remove = samp_probs < torch.topk(samp_probs, k)[0][..., -1, None]
        samp_probs[indices_to_remove] = 0
        if m is not None:
            samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
            samp_probs.mul_(1-m)
            samp_probs.add_(probs.mul(m))
        next_tokens = samp_probs.multinomial(1)
        next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()

    elif p is not None: 
        sorted_probs, sorted_indices = torch.sort(samp_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        sorted_samp_probs = sorted_probs.clone()
        sorted_samp_probs[sorted_indices_to_remove] = 0
        if m is not None:
            sorted_samp_probs.div_(sorted_samp_probs.sum(1).unsqueeze(1))
            sorted_samp_probs.mul_(1-m)
            sorted_samp_probs.add_(sorted_probs.mul(m))
        sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
        next_tokens = sorted_indices.gather(1, sorted_next_indices)
        next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()

    else:
        if m is not None:
            samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
            samp_probs.mul_(1-m)
            samp_probs.add_(probs.mul(m))
        next_tokens = samp_probs.multinomial(1)
        next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()
    return next_tokens.squeeze(1), next_logprobs

def get_batch_size(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.size(0)
    else:
        b_s = x[0].size(0)
    return b_s


def get_device(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.device
    else:
        b_s = x[0].device
    return b_s

def beam_search(model, visual: TensorOrSequence, max_len: int, eos_idx: int, beam_size: int, out_size=1, start_word=None,
                return_probs=False, **kwargs):
    bs = BeamSearch(model, max_len, eos_idx, beam_size, bos_idx=start_word)
    return bs.apply(visual, out_size, return_probs, **kwargs)
