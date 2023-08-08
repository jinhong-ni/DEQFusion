import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import BertEncoder, MLP, Linear
from datasets.imdb.get_data import get_dataloader
from training_structures.unimodal import train, test

import argparse

parser = argparse.ArgumentParser(description='DEQ Feature Fusion')
parser.add_argument('-p', '--dataset_path', required=True, type=str, help='Path to dataset')
args = parser.parse_args()

encoderfile = "unimodal_encoder_text_bert.pt"
headfile = "unimodal_head_text_bert.pt"
traindata, validdata, testdata = get_dataloader(
    os.path.join(args.dataset_path, 'multimodal_imdb.hdf5'), "../video/mmimdb", vgg=True, batch_size=16, metadata=os.path.join(args.dataset_path, 'metadata.npy'), use_bert=True)

encoders = BertEncoder().cuda()
# head = MLP(768, 512, 23).cuda()
head = Linear(768, 23).cuda()

train(encoders, head, traindata, validdata, 1000, early_stop=True, task="multilabel", save_encoder=encoderfile, modalnum=0,
      save_head=headfile, optimtype=torch.optim.AdamW, lr=5e-5, weight_decay=1e-5, criterion=torch.nn.BCEWithLogitsLoss(), use_bert=True, schedulertype=torch.optim.lr_scheduler.ReduceLROnPlateau)

print("Testing:")
encoder = torch.load(encoderfile).cuda()
head = torch.load(headfile).cuda()
test(encoder, head, testdata, "imdb",
     "unimodal_text", task="multilabel", modalnum=0, use_bert=True)
