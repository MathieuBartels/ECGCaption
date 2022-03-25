import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import GroupShuffleSplit

from ecgnet.utils.transforms import ToTensor, ApplyGain, Resample
from ecgnet.utils.dataset import UniversalECGDataset

from EcgCaptionGenerator.utils.pretrain import pre_train
from EcgCaptionGenerator.utils.dataset import collate_fn, CaptionDataset

import numpy as np
import lorem

from EcgCaptionGenerator.utils.vocab import Vocabulary
from nltk.tokenize import RegexpTokenizer
from collections import Counter

class FakeDataset():
    def __init__(self, length, topic, vocab, transform):
        self.length = length
        self.topic = topic
        self.transform = transform
        self.tokenizer = RegexpTokenizer('\d+\.?,?\d+|-?/?\w+-?/?\w*|\w+|\d+|<[A-Z]+>')

        if vocab is None:
            self.vocab = self.setup_vocab(0)
        else:
            self.vocab = vocab

    def setup_vocab(self, threshold):
        corpus = " ".join([lorem.sentence() for _ in range(self.length)])

        counter = Counter(self.tokenizer.tokenize(corpus))
        del counter['']

        words = [word for word, cnt in counter.items() if cnt >= threshold]

        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        # Add the words to the vocabulary.
        for _, word in enumerate(words):
            vocab.add_word(word)
        return vocab

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        waveform = np.random.rand(12, 5000)

        sample = {'waveform': waveform,
            'samplebase': 500,
            'gain': 4.88,
            'id': int(np.random.rand(1)),
        }

        
       
        sentence = lorem.sentence()

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
            topic_label_classes = 100

            topic_labels_bools = np.random.randint(2, size=topic_label_classes)
            topic_tensor = torch.from_numpy(topic_labels_bools).float()
            topic_tensor_norm = topic_tensor / topic_tensor.sum()

            sample['extra_label'] = topic_tensor_norm

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_loaders_toy_data(params, topic=False):
    vocab = None
    transform = transforms.Compose([Resample(500), ToTensor(), ApplyGain()])
    train_set = FakeDataset(1000, topic, vocab, transform)
    vocab = train_set.vocab
    val_set = FakeDataset(200, topic, vocab, transform)
    test_set = FakeDataset(200, topic, vocab, transform)

    train_loader = DataLoader(train_set, batch_size=params['batch_size'],
                              num_workers=4, collate_fn=collate_fn, shuffle=True)
    
    val_loader = DataLoader(val_set, batch_size=params['batch_size'],
                            num_workers=4, collate_fn=collate_fn)

    test_loader = DataLoader(test_set, batch_size=params['batch_size'],
                             num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab

def get_loaders(dev, params, topic):
    transform = transforms.Compose([Resample(500), ToTensor(), ApplyGain()])
    
    train_df = pd.read_csv(params['train_labels_csv'], index_col=0)

    train_inds, val_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2).split(train_df, groups=train_df['PseudoID']))

    train_dump = None #if dev else f"{params['train_labels_csv'][:-4]}.dump" 
    test_dump = None #if dev else f"{params['test_labels_csv'][:-4]}.dump"
    
    threshold, is_train, vocab = 15, True, None
    trainset = CaptionDataset(threshold, is_train, vocab, train_dump, topic,
                              'umcu', params['data_dir'], train_df.iloc[train_inds], transform=transform)

    is_train, vocab = False, trainset.vocab
    valset = CaptionDataset(threshold, is_train, vocab, train_dump, topic,
                            'umcu', params['data_dir'], train_df.iloc[val_inds], 
                            transform=transform)

    testset_df = pd.read_csv(params['test_labels_csv'], index_col=0)
    testset = CaptionDataset(threshold, is_train, vocab, test_dump, topic,
                             'umcu', params['data_dir'], testset_df, 
                             transform=transform)    

    train_loader = DataLoader(trainset, batch_size=params['batch_size'],
                              num_workers=4, collate_fn=collate_fn, shuffle=True)
    
    val_loader = DataLoader(valset, batch_size=params['batch_size'],
                            num_workers=4, collate_fn=collate_fn)

    test_loader = DataLoader(testset, batch_size=params['batch_size'],
                             num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab
