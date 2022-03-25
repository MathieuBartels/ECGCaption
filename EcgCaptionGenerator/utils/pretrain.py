
from pytorch_lightning.loggers.neptune import NeptuneLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def pre_train(system, dataloader, testloader, params):
    model = system(**params)

    api_key = open("api_key.txt", "r").read()
    neptune_logger = NeptuneLogger(api_key=api_key,
        project_name="mathieubartels/pre-train",
        params=params,
        experiment_name='pre-train')
        # upload_source_files=['*.py', '*.json', '../ecgnet/**/*.py'])

    early_stopping = EarlyStopping('val_loss', mode='min', patience=3)

    checkpoint_callback = ModelCheckpoint(save_last=True)

    trainer = pl.Trainer(max_epochs=params['epochs'], 
                        #  fast_dev_run=True,
                         default_root_dir='/training/captioning/models/',
                         logger=neptune_logger,
                         log_every_n_steps=5,
                         callbacks=[early_stopping, checkpoint_callback],
                         gpus=1)
    
    trainer.fit(model, dataloader, testloader)

    return model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).model
    
if __name__ == '__main__':
    import json
    import pandas as pd
    import sys
    sys.path.append('..')
    from torchvision import transforms
    from torch.utils.data import DataLoader

    from ecgnet.utils.transforms import ToTensor, ApplyGain, Resample
    from ecgnet.utils.dataset import UniversalECGDataset
    from ecgnet.systems.triage_uncertainty import TriageSystem

    params = json.load(open('config.json', 'r'))
    
    transform = transforms.Compose([Resample(500), ToTensor(), ApplyGain()])

    pre_train_df = pd.read_csv(params['pretrain_labels_csv'], index_col=0)
    pre_train_dataset = UniversalECGDataset('umcu', params['data_dir'], pre_train_df, transform=transform)
    pre_train_loader = DataLoader(pre_train_dataset, batch_size=64,
                            num_workers=0, shuffle=True)

    pre_test_df = pd.read_csv(params['pretest_labels_test_csv'], index_col=0)
    pre_test_dataset = UniversalECGDataset('umcu', params['data_dir'], pre_test_df, transform=transform)
    pre_test_loader = DataLoader(pre_test_dataset, batch_size=64,
                            num_workers=0, shuffle=False)

    ecgres_net = pre_train(TriageSystem, pre_train_loader, pre_test_loader, params)
    pre_train(params)


