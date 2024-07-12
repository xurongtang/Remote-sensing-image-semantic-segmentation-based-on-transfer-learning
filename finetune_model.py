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

def freeze_encoder(model):
    for name, param in model.named_parameters():
        # print(name)
        if 'encoder' in name:
            param.requires_grad = False

def initialize_decoder(model):
    for name, module in model.named_modules():
        if 'decoder' in name:
            for layer in module.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

def finetune(model):
    # 冻结encoder部分
    freeze_encoder(model)
    # 随机初始化decoder部分
    initialize_decoder(model)

def verify_freeze(model, freeze_layer_keyword):
    for name, param in model.named_parameters():
        if freeze_layer_keyword in name:
            print(f"Layer {name} frozen: {not param.requires_grad}")

def main(model_name,encoder_name,finetune_epoch,pre_trained_modelpath,save_modelpath):
    
    root_train = '/home/rton/pan1/CG_dataset/train/'
    root_val = '/home/rton/pan1/CG_dataset/val/'

    train_dataset = segmentation_dataset(root_train, "train")
    valid_dataset = segmentation_dataset(root_val, "val")

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=n_cpu)

    model = Model(model_name,encoder_name=encoder_name, in_channels=3, out_classes=1,finetune=True)
    model.load_state_dict(torch.load(pre_trained_modelpath))

    finetune(model)
    # verify_freeze(model,'encoder')

    # train
    trainer = pl.Trainer(gpus=1, max_epochs=finetune_epoch)
    trainer.fit(model,train_dataloaders=train_dataloader,val_dataloaders=valid_dataloader)
    
    torch.save(model.state_dict(),save_modelpath)

    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--encoder_name",type=str)
    parser.add_argument("--finetune_epoch",type=int,default=3)
    parser.add_argument("--pre_trained_modelpath",type=str)
    parser.add_argument("--model_savepath", type=str)
    args = parser.parse_args()
    print('finetune...\n',args)
    main(args.model_name,
         args.encoder_name,
         args.finetune_epoch,
         args.pre_trained_modelpath,
         args.model_savepath)