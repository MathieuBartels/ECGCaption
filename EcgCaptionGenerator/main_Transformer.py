import sys
import json
import pytorch_lightning as pl

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load

sys.path.append('..')
from EcgCaptionGenerator.systems.transformer import Transformer
from EcgCaptionGenerator.systems.topic import Topic
from EcgCaptionGenerator.systems.topic_unchanged_decoder import TopicSimDecoder
from EcgCaptionGenerator.util import get_encoder, get_loaders
import pandas as pd

def cli_main(params, dev):
    pl.seed_everything(1234)
    # api_key = open("api_key.txt", "r").read()

    neptune_logger = DummyLogger() 
    # if dev else NeptuneLogger(api_key=api_key,
    #     project_name="mathieubartels/transformer",
    #     params=params,
    #     experiment_name='transformer',
    #     upload_source_files=['*.py', '*.json', '../EcgCaptionGenerator/**/*.py'])

    train_loader, val_loader, test_loader, vocab = get_loaders(dev, params, topic=False)
    
    model = Transformer(vocab, **params)

    # if params['continuetrain']:
    #     model = Transformer.load_from_checkpoint(params['continuetrain'])
    if params['pre_train']:
        pre_trained = TopicSimDecoder.load_from_checkpoint(params['pre_train'])
        # pre_trained = Topic.load_from_checkpoint(params['pretrain'])
        model.load_pre_trained(pre_trained)

    early_stopping = EarlyStopping('val_loss', mode='min', patience=6)
    checkpoint_callback = ModelCheckpoint(save_last=True)

    trainer = pl.Trainer(max_epochs=params['epochs'], 
                        #  fast_dev_run=True,
                         default_root_dir='/training/transformer/models/',
                         logger=neptune_logger,
                         log_every_n_steps=5,
                         callbacks=[early_stopping, checkpoint_callback],
                         gpus=1)

    trainer.fit(model, train_loader, val_loader)

    test_df = pd.read_csv(params['test_labels_csv'], index_col=0)
    gts = test_df.apply(lambda x: {x['TestID']: [x['Label']]}, axis=1).to_list()
    gts = {list(dict_item.keys())[0]: list(dict_item.values())[0] for dict_item in gts}
    
    model.gts = gts
    trainer.test(test_dataloaders=test_loader)

if __name__ == '__main__':
    # params, dev = json.load(open('config_transformer_muse.json', 'r')), False
    params, dev = json.load(open('config_transformer_consult.json', 'r')), False
    # params, dev = json.load(open('config_transformer_debug.json', 'r')), True
    
    cli_main(params, dev)
