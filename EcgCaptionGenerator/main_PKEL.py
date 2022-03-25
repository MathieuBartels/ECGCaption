import sys
import json
import pytorch_lightning as pl

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load

sys.path.append('..')
from EcgCaptionGenerator.systems.topic_unchanged_decoder import TopicSimDecoder
from EcgCaptionGenerator.systems.topic import Topic
from EcgCaptionGenerator.util import get_loaders,get_loaders_toy_data
import pandas as pd

def cli_main(params, dev):
    pl.seed_everything(1234)
    # api_key = open("api_key.txt", "r").read()

    neptune_logger = neptune_logger = DummyLogger() #NeptuneLogger(api_key=api_key,
        # project_name="topic",
        # params=params,
        # experiment_name='topic',
        # upload_source_files=['*.py', '*.json', '../EcgCaptionGenerator/**/*.py'])

    # train_loader, val_loader, test_loader, vocab = get_loaders(dev, params, topic=True)
    train_loader, val_loader, test_loader, vocab = get_loaders_toy_data(params, topic=True)
    
    model = TopicSimDecoder(vocab, **params)
    if params['pre_train']:
        # model = Topic(vocab, **params)
        pre_trained = TopicSimDecoder.load_from_checkpoint(params['pre_train'])

        model.pre_train = True
        model.load_pre_trained(pre_trained)

    early_stopping = EarlyStopping('val_loss', mode='min', patience=6)
    checkpoint_callback = ModelCheckpoint(save_last=True)

    trainer = pl.Trainer(max_epochs=params['epochs'], 
                        #  fast_dev_run=True,
                         default_root_dir='./training/captioning/models/',
                         logger=neptune_logger,
                         log_every_n_steps=5,
                         gradient_clip_val=1,
                         callbacks=[early_stopping, checkpoint_callback],
                         gpus=0)

    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)

if __name__ == '__main__':
    params, dev = json.load(open('config_PKEL.json', 'r')), False
    # params, dev = json.load(open('config_debug.json', 'r')), True
    
    cli_main(params, dev)
