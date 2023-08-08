import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import Linear, ResNet152
from datasets.imdb.get_data import get_dataloader
from training_structures.unimodal import train, test

import argparse

parser = argparse.ArgumentParser(description='DEQ Feature Fusion')
parser.add_argument('-p', '--dataset_path', required=True, type=str, help='Path to dataset')
args = parser.parse_args()

encoderfile = "unimodal_encoder_image_resnet152.pt"
headfile = "unimodal_image_resnet152.pt"
traindata, validdata, testdata = get_dataloader(
    os.path.join(args.dataset_path, 'mmimdb'), "../video/mmimdb", vgg=False, batch_size=64, metadata=os.path.join(args.dataset_path, 'metadata.npy'), use_raw=True)


encoders = ResNet152(out_dim=768).cuda()
head = Linear(768, 23).cuda()

train(encoders, head, traindata, validdata, 100, early_stop=True, task="multilabel", save_encoder=encoderfile, modalnum=1,
      save_head=headfile, optimtype=torch.optim.AdamW, lr=1e-4, weight_decay=1e-6, criterion=torch.nn.BCEWithLogitsLoss(), 
      schedulertype=torch.optim.lr_scheduler.ReduceLROnPlateau, freeze_epoch=3)

print("Testing:")
encoder = torch.load(encoderfile).cuda()
head = torch.load(headfile).cuda()
test(encoder, head, testdata, "imdb",
     "unimodal_image", task="multilabel", modalnum=1)
