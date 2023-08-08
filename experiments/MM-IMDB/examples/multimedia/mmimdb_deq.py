import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import Linear, MaxOut_MLP, MLP
from datasets.imdb.get_data import get_dataloader
from fusions.DEQ_fusion import DEQFusion
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test

import argparse

parser = argparse.ArgumentParser(description='DEQ Feature Fusion')
parser.add_argument('-p', '--dataset_path', required=True, type=str, help='Path to dataset')
args = parser.parse_args()


filename = "best_deq.pt"
traindata, validdata, testdata = get_dataloader(
    # "/home/nijinhong1/deq/mmimdb/multimodal_imdb.hdf5", "../video/mmimdb", vgg=True, batch_size=256)
    os.path.join(args.dataset_path, 'multimodal_imdb.hdf5'), "../video/mmimdb", vgg=True, batch_size=256, metadata=os.path.join(args.dataset_path, 'metadata.npy'))

encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False),
            MaxOut_MLP(512, 1024, 4096, 512, False)]
# encoders = [
#     MLP(300, 512, 512),
#     MLP(4096, 1024, 512),
# ]
head = Linear(512, 23).cuda()
fusion = DEQFusion(512, 2, 105, 106, 'abs').cuda()
# fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel",
      save=filename, optimtype=torch.optim.AdamW, lr=1e-3, weight_decay=1e-2, objective=torch.nn.BCEWithLogitsLoss(), lower_lr_for_fusion=True)

print("Testing:")
model = torch.load(filename).cuda()
# model.fuse.f_thres = 150
# model.fuse.b_thres = 151
test(model, testdata, method_name="deq", dataset="imdb",
     criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")
