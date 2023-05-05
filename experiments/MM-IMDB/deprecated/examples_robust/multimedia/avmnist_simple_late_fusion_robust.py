from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data_robust import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Simple_Late_Fusion import train, test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

traindata, validdata, testdata, robustdata = get_dataloader(
    '../../../../yiwei/avmnist/_MFAS/avmnist')
channels = 6
encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
head = MLP(channels*40, 100, 10).cuda()

fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 30, optimtype=torch.optim.SGD,
      lr=0.1, weight_decay=0.0001, save='avmnist_simple_late_fusion_best.pt')

model = torch.load('avmnist_simple_late_fusion_best.pt').cuda()
print("Testing:")
test(model, robustdata)

print("Robustness testing:")
test(model, testdata)
