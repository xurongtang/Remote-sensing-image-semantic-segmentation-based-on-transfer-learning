import os
import torch
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pprint import pprint
from torch.utils.data import DataLoader
from get_data.data_provide import segmentation_dataset
from model.base_model import Model
import sys
import ssl
import argparse
torch.multiprocessing.set_sharing_strategy('file_system')
ssl._create_default_https_context = ssl._create_unverified_context


def main(model_name,encoder_name,epoch,load_model,model_savepath):

    root_train = '/home/rton/pan1/competetion_data/train/'
    root_val = '/home/rton/pan1/competetion_data/val/'

    # root_train = '/home/rton/pan1/CG_dataset/train/'
    # root_val = '/home/rton/pan1/CG_dataset/val/'

    train_dataset = segmentation_dataset(root_train, "train")
    valid_dataset = segmentation_dataset(root_val, "val")

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=n_cpu)

    model_name = model_name
    encoder_name = encoder_name
    epoch = int(epoch)

    model = Model(model_name,encoder_name=encoder_name, in_channels=3, out_classes=1)
    if load_model is not None:
        model.load_state_dict(torch.load(load_model))

    # train
    trainer = pl.Trainer(gpus=1, max_epochs=epoch)
    trainer.fit(model,train_dataloaders=train_dataloader,val_dataloaders=valid_dataloader)

    torch.save(model.state_dict(),model_savepath)

    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--encoder_name",type=str)
    parser.add_argument("--epoch",type=int)
    parser.add_argument("--load_model",type=str,default=None)
    parser.add_argument("--model_savepath", type=str)
    args = parser.parse_args()
    print('pre-trained...\n',args)
    main(args.model_name,
         args.encoder_name,
         args.epoch,
         args.load_model,
         args.model_savepath)