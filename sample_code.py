import torch
from torch import nn
import torchvision
import torchvision.transforms as T
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc
import numpy as np
import pandas as pd
import time
import math
import os
import argparse
import json
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet

from pathlib import Path

from typing import List
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import ToDevice, ToTensor, ToTorchImage, Convert, NormalizeImage

SEED = 42


class enetb6(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        print("Using EfficientNet-b6 as the backbone")
        self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)

        self.model._swish = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.model._swish(x)
        return x


class resnet101(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        print("Using Resnet-101 as the backbone")
        self.model = torchvision.models.resnet101(pretrained=True)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x


def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    device = None
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
        device = torch.device("cpu")
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    if not device:
        device = torch.device("cuda")
    device_ids = list(range(n_gpu_use))
    return device, device_ids


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.manual_seed(SEED)


def cleanup():
    torch.distributed.destroy_process_group()


def compute_performance_metrics(outputs_label, target_label, num_classes):

    outputs_label, target_label = outputs_label.cpu().numpy(), target_label.cpu().numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mean_roc_auc = 0.0
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(target_label[:, i], outputs_label[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        if math.isnan(roc_auc[i]):
            # roc_auc[i] = previous_auroc[i]
            print("math.isnan while calculating auroc")
        mean_roc_auc += roc_auc[i]
    mean_roc_auc /= num_classes
    return roc_auc, mean_roc_auc


def train_loop(dataloader, model, optimizer, rank, config, epoch):
    train_loss = []
    model.train()

    for i, train_batch in enumerate(dataloader):
        print("iteration number: ", epoch*len(dataloader)+i)
        x, y = train_batch

        outputs_cls = model(x)
        loss = F.binary_cross_entropy(outputs_cls, y)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Training Epoch End
    train_loss = torch.mean(torch.tensor(train_loss))
    print("avg_train_loss: ", train_loss.item())


def validation_loop(val_loader, model, optimizer, rank, config, epoch, best_auc):
    model.eval()

    with torch.no_grad():
        val_loss = []
        target_labels = []
        predicted_labels = []

        for i, val_batch in enumerate(val_loader):
            print("validation batch number: ", epoch*len(val_loader)+i)
            x, y = val_batch

            outputs_cls = model(x)
            loss = F.binary_cross_entropy(outputs_cls, y).item()

            val_loss.append(loss)
            target_labels.append(y)
            predicted_labels.append(outputs_cls)

        # Validation Epoch End
        val_loss = torch.mean(torch.tensor(val_loss))
        print("avg_val_loss: ", val_loss.item())

        targetLabels = torch.cat([x for x in target_labels])
        predictedLabels = torch.cat([x for x in predicted_labels])
        val_auroc_classes, mean_roc_auc = compute_performance_metrics(predictedLabels.detach(), targetLabels.detach(),
                                                                      config.num_classes)

        if rank == 0:
            if mean_roc_auc > best_auc:
                best_auc = mean_roc_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_validation_loss': val_loss,
                    'best_auroc': best_auc,
                    'individual_classes_auroc': val_auroc_classes,
                    'batch_size': config.bs,
                    'learning_rate': config.lr,
                    'optimizer': config.optimizer
                    }, config.PATH)

    return best_auc, val_loss


def train_model(train_loader, val_loader, model, optimizer, rank, config):
    """ training of a model, same for teacher and student model

    Args:
        train_loader ([dataloader]): dataloader for training
        val_loader ([dataloader]): dataloader for validation
        model ([type]): model for generating predictions
        optimizer([])
        rank([])

    Returns:
        [type]:
    """

    best_auc = 0.0
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    for epoch in range(config.epochs):
        print("Epoch: ", epoch)
        train_loop(train_loader, model, optimizer, rank, config, epoch)
        best_auc, val_loss = validation_loop(val_loader, model, optimizer, rank, config, epoch, best_auc)
        if config.scheduler:
            scheduler.step()

    if rank == 0:
        print('Best Auroc: {:4f}'.format(best_auc))


def ddp_trainer(gpu, config):
    print('starting student training on multiple gpus')
    config.rank = gpu
    setup(config.rank, config.world_size, config.port)
    torch.cuda.set_device(config.rank)

    if config.model == 'resnet101':
        model = resnet101(config.num_classes)
        if config.channels_last:
            model = model.to(memory_format=torch.channels_last)
    elif config.model == 'enetb6':
        model = enetb6(config.num_classes)
        if config.channels_last:
            model = model.to(memory_format=torch.channels_last)

    model.cuda(config.rank)
    model = DDP(model, device_ids=[config.rank])

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.lr,
                                weight_decay=config.weight_decay,
                                momentum=config.momentum,
                                nesterov=True
                                )

    paths = {}
    paths['train_beton_path'] = config.train_beton_path
    paths['val_beton_path'] = config.val_beton_path

    MEAN = np.array([128, 128, 128])
    STD = np.array([128, 128, 128])

    loaders = {}
    for name in ['train', 'val']:
        label_pipeline: List[Operation] = [NDArrayDecoder(), ToTensor(), ToDevice(config.rank)]

        if config.channels_last:
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder(), ToTensor(), ToDevice(config.rank), ToTorchImage(channels_last=True), NormalizeImage(MEAN, STD, np.float32)]
        else:
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder(), ToTensor(), ToDevice(config.rank), ToTorchImage(channels_last=False), Convert(torch.float32), torchvision.transforms.Normalize([128, 128, 128], [128, 128, 128])]

        # Create loaders
        loaders[name] = Loader(
            paths[f'{name}_beton_path'],
            batch_size=config.bs,
            num_workers=config.num_workers,
            order=OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL,
            distributed=(name == 'train'),
            seed=0,
            drop_last=(name == 'train'),
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            }
        )

    train_model(loaders['train'], loaders['val'], model, optimizer, config.rank, config)

    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_workers",            type=int,   default=None)
    parser.add_argument("--lr",                     type=float, default=None)
    parser.add_argument("--bs",                     type=int,   default=None)
    parser.add_argument("--epochs",                 type=int,   default=None)
    parser.add_argument("--gpus",                   type=int,   default=2)
    parser.add_argument("--optimizer",              type=str,   default='SGD')
    parser.add_argument("--weight_decay",           type=float, default=0)
    parser.add_argument("--momentum",               type=float, default=0.9)
    parser.add_argument('--scheduler',              dest='scheduler',         action='store_true', help='whether to use scheduler or not')
    parser.add_argument("--gamma",                  type=float, default=0.1,                       help="lr decay for stepLR")
    parser.add_argument("--step_size",              type=int,   default=None,                      help="step size for stepLR")
    parser.add_argument('--model',                  type=str,   default='',                        help='resnet101, enetb6')
    parser.add_argument('--port',                   type=int,   default=None,                      help='port for ddp training')
    parser.add_argument('--train_beton_path',       type=str,   default=None,                      help='beton file path for training data')
    parser.add_argument('--val_beton_path',         type=str,   default=None,                      help='beton file path for validation data')
    parser.add_argument('--channels_last',          dest='channels_last',     action='store_true', help='channels last format if specified else contiguous memory format')

    config = parser.parse_args()
    print(config)
    print("File Path: ", Path(__file__).absolute())

    device, device_ids = setup_device(config.gpus)
    config.world_size = len(device_ids)

    mp.spawn(ddp_trainer, nprocs=config.world_size, args=(config,), join=True)

    print("done")
