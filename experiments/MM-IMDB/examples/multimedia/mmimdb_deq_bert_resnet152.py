import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import Linear, MaxOut_MLP, MLP, BertEncoder, ResNet152
from datasets.imdb.get_data import get_dataloader
from fusions.DEQ_fusion import DEQFusion
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test

from pytorch_pretrained_bert import BertAdam

import argparse

parser = argparse.ArgumentParser(description='DEQ Feature Fusion')
parser.add_argument('-p', '--dataset_path', required=True, type=str, help='Path to dataset')
args = parser.parse_args()


# filename = "best_deq_bert_resnet152_lr_patience3_1e-4_1e-6_gas10_adamw_freeze10.pt"
filename = "BEST_BACKUP_deq_bert_resnet152_lr_patience3_1e-4_1e-6_gas10_adamw_freeze01.pt"
# filename = "best_deq_bert_resnet152_lr_patience3_1e-4_1e-6_gas10_adamw_freeze11_normalized.pt"
traindata, validdata, testdata = get_dataloader(
    # "/home/nijinhong1/deq/mmimdb/multimodal_imdb.hdf5", "../video/mmimdb", vgg=True, batch_size=256)
    os.path.join(args.dataset_path, 'mmimdb'), "../video/mmimdb", vgg=False, batch_size=16, metadata=os.path.join(args.dataset_path, 'metadata.npy'), use_bert=True, use_raw=True)

# encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False),
#             MaxOut_MLP(512, 1024, 4096, 512, False)]
encoders = [BertEncoder().cuda(),
            ResNet152(out_dim=768, normalized=False).cuda()]
# encoders = [
#     MLP(300, 512, 512),
#     MLP(4096, 1024, 512),
# ]
head = Linear(768, 23).cuda()
fusion = DEQFusion(768, 2, 105, 106, 'abs').cuda()

freqs = [traindata.dataset.label_freqs[l] for l in traindata.dataset.labels]
label_weights = (torch.FloatTensor(freqs) / len(traindata)) ** -1
# print(label_weights)
# fusion = Concat().cuda()

# train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel",
#       save=filename, optimtype=torch.optim.AdamW, lr=1e-3, weight_decay=1e-2, objective=torch.nn.BCEWithLogitsLoss(), use_bert=True)
train(encoders, fusion, head, traindata, validdata, 100, early_stop=True, task="multilabel",
      save=filename, optimtype=torch.optim.AdamW, lr=1e-4, weight_decay=1e-6, objective=torch.nn.BCEWithLogitsLoss(), use_bert=True, 
      schedulertype=torch.optim.lr_scheduler.ReduceLROnPlateau, gradient_accumulation_steps=10, freeze_img_epoch=0, freeze_txt_epoch=1)

print("Testing:")
model = torch.load(filename).cuda()
# model.fuse.f_thres = 150
# model.fuse.b_thres = 151
test(model, testdata, method_name="deq", dataset="imdb",
     criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel", use_bert=True)
