import sys
import json
import pytorch_lightning as pl

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load

sys.path.append('..')
print(sys.path)
from EcgCaptionGenerator.systems.topic_transformer import TopicTransformer
# from EcgCaptionGenerator.systems.topic_transformer import TopicTransformer
from EcgCaptionGenerator.systems.topic import Topic
from EcgCaptionGenerator.systems.topic_unchanged_decoder import TopicSimDecoder
from EcgCaptionGenerator.util import get_loaders, get_loaders_toy_data
import pandas as pd

def cli_main(params, dev):
    pl.seed_everything(1234)
    # api_key = open("api_key.txt", "r").read()

    neptune_logger = DummyLogger() #if dev else NeptuneLogger(api_key=api_key,
        # project_name=projname,
        # params=params,
        # experiment_name='tansformertopic',
        # upload_source_files=['*.py', '*.json', '../EcgCaptionGenerator/**/*.py'])

    # train_loader, val_loader, test_loader, vocab = get_loaders(dev, params, topic=True)
    train_loader, val_loader, test_loader, vocab = get_loaders_toy_data(params, topic=True)
    

    model = TopicTransformer(vocab, **params)

    # if params['continuetrain']:
    #     model = Transformer.load_from_checkpoint(params['continuetrain'])
    if params['pre_train']:
        pre_trained = TopicTransformer.load_from_checkpoint(params['pre_train'])
        model.load_pre_trained(pre_trained)

    early_stopping = EarlyStopping('val_loss', mode='min', patience=6)
    checkpoint_callback = ModelCheckpoint(save_last=True)

    trainer = pl.Trainer(max_epochs=params['epochs'], 
                        #  fast_dev_run=True,
                         default_root_dir='./training/transformertopic/models/',
                         logger=neptune_logger,
                         log_every_n_steps=5,
                         callbacks=[early_stopping, checkpoint_callback],
                         gpus=0)

    trainer.fit(model, train_loader, val_loader)

    # test_df = pd.read_csv(params['test_labels_csv'], index_col=0)
    # gts = test_df.apply(lambda x: {x['TestID']: [x['Label']]}, axis=1).to_list()
    # gts = {list(dict_item.keys())[0]: list(dict_item.values())[0] for dict_item in gts}
    
    # model.gts = gts
    trainer.test(test_dataloaders=test_loader)

if __name__ == '__main__':
    params, dev = json.load(open('config_PKT.json', 'r')), False

    
    cli_main(params, dev)
