import sys
sys.path.append('..')
from operator import itemgetter

from torchvision import transforms
import torch
import pandas as pd
import numpy as np

from nltk.tokenize import RegexpTokenizer
from collections import Counter
from torch.utils.data import DataLoader

from ecgnet.utils.dataset import UniversalECGDataset
from ecgnet.utils.transforms import ToTensor, ApplyGain, Resample

from EcgCaptionGenerator.utils.vocab import Vocabulary
import pickle

class CaptionDataset(UniversalECGDataset):
    def __init__(self, threshold, train, vocab, dump_location, topic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = RegexpTokenizer('\d+\.?,?\d+|-?/?\w+-?/?\w*|\w+|\d+|<[A-Z]+>')
        if train:
            self.vocab = self.setup_vocab(self.dataset['Label'], threshold)
        else:
            self.vocab = vocab

        self.topic = topic

        self.waveform_dump = None
        if dump_location:
            with open(dump_location, 'rb') as f:
                self.waveform_dump = pickle.load(f) 

    def __getitem__(self, idx):
        if self.waveform_dump:
            sample_id = self.dataset['TestID'].iloc[idx]
            waveform = self.waveform_dump[sample_id]
        else:
            waveform, sample_id = self.get_waveform(idx)

        # Add waveform, original sample base, gain and ID to sample
        sample = {
            'waveform': waveform,
            'samplebase': int(self.dataset['SampleBase'].iloc[idx]),
            'gain': float(self.dataset['Gain'].iloc[idx]),
            'id': sample_id,
        }

        # Sometimes additional information is needed (e.g. for a median cutoff)
        possible_cols = ['AcqDate', 'POnset', 'TOffset', 'VentricularRate',
                         'QOnset', 'POffset', 'QOffset', 'start_idx',
                         'end_idx']

        for col in possible_cols:
            if col in self.dataset:
                sample[col.lower()] = self.dataset[col].iloc[idx]

        if self.label in self.dataset.columns.values:
            sentence = self.dataset[self.label].iloc[idx]
            try:
                tokens = self.tokenizer.tokenize(sentence)
            except:
                print(sentence)
                raise Exception()
            vocab = self.vocab
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)

            sample['label'] = target

        if self.topic:
            topic_labels_columns_contains = ['class', 'rhythm', 'conduction', 'acuteischemia', 'possibleischemia',
            'unclearischemia', 'oldischemia', 'noischemia', 'axis', 'other', 'quality']

            topic_labels_bools = self.dataset.columns.str.contains('|'.join(topic_labels_columns_contains))
            topic_labels_columns = self.dataset.columns[topic_labels_bools]
            topic_tensor = torch.from_numpy(self.dataset[topic_labels_columns].iloc[idx].values).float()
            topic_tensor_norm = topic_tensor / topic_tensor.sum()

            sample['extra_label'] = topic_tensor_norm


        if self.transform:
            sample = self.transform(sample)
        return sample

    def setup_vocab(self, labels, threshold):
        corpus = labels.str.cat(sep=" ")

        counter = Counter(self.tokenizer.tokenize(corpus))
        del counter['']

        words = [word for word, cnt in counter.items() if cnt >= threshold]

        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            vocab.add_word(word)
        return vocab

def collate_fn(data):
    """Creates mini-batch tensors from the dicts (waveform, samplebase, gain, id, captions).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of dict (waveform, samplebase, gain, id, captions). 

    Returns:
    """
    captions = [d['label'] for d in data]
    lengths = [len(cap) for cap in captions]
    
    if len(lengths) == 1:
        return data[0]['waveform'].unsqueeze(0), data[0]['samplebase'], data[0]['gain'], data[0]['id'], data[0]['label'].unsqueeze(0).long(), lengths, data[0]['extra_label'].unsqueeze(0)

    ind = np.argsort(lengths)[::-1]

    lengths = list(itemgetter(*ind)(lengths)) 
    captions = list(itemgetter(*ind)(captions))

    waveforms = list(itemgetter(*ind)([d['waveform'] for d in data]))
    samplebases = list(itemgetter(*ind)([d['samplebase'] for d in data]))
    gains = list(itemgetter(*ind)([d['gain'] for d in data]))
    ids = list(itemgetter(*ind)([d['id'] for d in data]))
    
    # Merge images (from tuple of 3D tensor to 4D tensor).
    waveforms = torch.stack(waveforms, 0)
    

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    if 'extra_label' in data[0]:
        extra_label = list(itemgetter(*ind)([d['extra_label'] for d in data]))
        extra_label = torch.stack(extra_label, 0)
        return waveforms, samplebases, gains, ids, targets, lengths, extra_label

    return waveforms, samplebases, gains, ids, targets, lengths

def indexto1hot(vocab_len, index):
    one_hot = np.zeros([vocab_len])
    one_hot[index] = 1
    return one_hot

def test_dataset():
    params = {
        'train_labels_csv': '/training/captioning/datasets/train_muse.csv',
        'data_dir': '/raw_data/umcu_rhythm'
    }

    train_df = pd.read_csv(params['train_labels_csv'], index_col=0).iloc[:30]
    
    transform = transforms.Compose([ToTensor(), ApplyGain(), Resample(500)])

    threshold = 5
    is_Train = True
    vocab = None
    trainset = CaptionDataset(threshold, is_Train, vocab, 'umcu', params['data_dir'], train_df,
                                transform=transform)
     
    train_loader = DataLoader(trainset, batch_size=6, num_workers=8, collate_fn=collate_fn)
    for sample in train_loader:
        print(sample)
        break

# test_dataset()