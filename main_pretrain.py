# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    
    # BT
    parser.add_argument('--add_global', default=0, type=int,
                        help='add_global')
    parser.add_argument('--eval_global', default=0, type=int,
                        help='eval_global')
    parser.add_argument('--dropout', default=0., type=float,
                        help='dropout')
    parser.add_argument('--exp_id', default="", type=str,
                        help='exp_id')
    parser.add_argument('--decay_drop_bt', default=0., type=float,
                        help='drop_patch')
    parser.add_argument('--drop_patch', default=0., type=float,
                        help='drop_patch')
    parser.add_argument('--shuffle_patch', action='store_true', )
    parser.add_argument('--no_fp16_bt', default=0, type=int,
                        help='no_fp16_bt')
    parser.add_argument('--num_heads', default=4, type=int,
                        help='num_heads')
    parser.add_argument('--start_idx', default=8, type=int,
                        help='start_idx: start bt layer index')
    parser.add_argument('--start_bt_epoch', default=0, type=int,
                        help='start_bt_epoch: start bt layer index')
    parser.add_argument('--insert_idx', action='append', type=int,
                        help='insert idx list')
    parser.add_argument('--small_seq', action='store_true',)
    parser.add_argument('--all_patches', action='store_true',)
    parser.add_argument('--drop_path_bt', default=0., type=float)
    parser.add_argument('--not_cls_token', action='store_true'),
    parser.add_argument('--cls_token_only', action='store_true'),
    parser.add_argument('--shared_bt', default=1, type=int,),
    parser.add_argument('--empty_bt', default=0, type=int,),
    parser.add_argument('--add_norm_bt', default=1, type=int,),
    parser.add_argument('--add_mlp_bt', default=0, type=int,),
    parser.add_argument('--mlp_enc', default=0, type=int,),
    parser.add_argument('--no_grad_bt', default=0, type=int,),
    parser.add_argument('--bt_decay', default=0., type=float,
                        help='drop_patch')
    parser.add_argument('--mlp_decay', default=0., type=float,
                       help='drop_patch')
    parser.add_argument('--bt_lr', default=0.5, type=float,)
    parser.add_argument('--skip_bt', action='store_true',)
    parser.add_argument('--pretrained', action='store_true',
                        help='pretrained')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    
    
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

from timm.models.vision_transformer import Attention, Block
class BlockWrap(torch.nn.Module):

    def __init__(self, block: Block, dim, args):
        super().__init__()
        self.block = block
        self.attn_bt = Attention(dim, num_heads=6, qkv_bias=0., attn_drop=0.5, proj_drop=args.drop)
        from timm.models.layers import DropPath
        self.drop_path_bt = DropPath(args.drop_path_bt) if args.drop_path_bt > 0. else torch.nn.Identity()
        self.norm3 = torch.nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x + self.block.drop_path(self.block.attn(self.block.norm1(x)))
        x = x + self.block.drop_path(self.block.mlp(self.block.norm2(x)))
        if self.training:
            old_x = x
            old_x = old_x.transpose(0, 1)
            x = x + self.drop_path_bt(self.norm3(self.attn_bt(old_x).transpose(0, 1)))
        # x = self.block(x)
        return x


class BlockBF(torch.nn.Module):

    def __init__(self, block: Block, fp16_type=0):
        super().__init__()
        self.block = block
        self.fp16_type = fp16_type

    def forward(self, x):
        from torch.cuda.amp import autocast
        if self.fp16_type == 2:

            with autocast(enabled=False):
                old_x = self.block.norm1(x)
                x = x + torch.transpose(self.block.drop_path(torch.transpose(self.block.attn(old_x.float()), 1, 0)), 1, 0)
                old_x = self.block.norm2(x)
            x = x + torch.transpose(self.block.drop_path(torch.transpose(self.block.mlp(old_x), 1, 0)), 1, 0)
        elif self.fp16_type == 0:
            x = x + torch.transpose(self.block.drop_path(torch.transpose(self.block.attn(self.block.norm1(x)), 1, 0)), 1, 0)
            x = x + torch.transpose(self.block.drop_path(torch.transpose(self.block.mlp(self.block.norm2(x)), 1, 0)), 1, 0)
        else:
            old_x = self.block.norm1(x)
            with autocast(enabled=False):
                x = x + torch.transpose(self.block.drop_path(torch.transpose(self.block.attn(old_x.float()), 1, 0)), 1, 0)
            x = x + torch.transpose(self.block.drop_path(torch.transpose(self.block.mlp(self.block.norm2(x)), 1, 0)), 1, 0)


        return x


class BlockWrap32(torch.nn.Module):

    def __init__(self, block: Block, fp16_type: int):
        super().__init__()
        self.block = block
        self.fp16_type = fp16_type

    def forward(self, x):
        from torch.cuda.amp import autocast
        if self.fp16_type == 2:

            with autocast(enabled=False):
                old_x = self.block.norm1(x)
                x = x + self.block.drop_path(self.block.attn(old_x.float()))
                old_x = self.block.norm2(x)
            x = x + self.block.drop_path(self.block.mlp(old_x))
        else:
            old_x = self.block.norm1(x)
            with autocast(enabled=False):
                x = x + self.block.drop_path(self.block.attn(old_x.float()))
            x = x + self.block.drop_path(self.block.mlp(self.block.norm2(x)))
        return x

class BlockWrapDebug(torch.nn.Module):

    def __init__(self, block: Block):
        super().__init__()
        self.block = block

    def forward(self, x):
        tmp_x = self.block.drop_path(self.block.attn(self.block.norm1(x)))
        for n,p in self.block.named_parameters():
            if n.__contains__('attn.qkv.weight'):
                print(n, p.detach().cpu().numpy(), p.grad)
        x = x + tmp_x
        tmp_x = self.block.drop_path(self.block.mlp(self.block.norm2(x)))

        x = x + tmp_x
        return x

class AttentionOnly(torch.nn.Module):

    def __init__(self, block: Block, drop_path = 0. , add_global=70):
        super().__init__()
        self.block = block
        from timm.models.layers import DropPath
        self.add_global=add_global
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        self.block.mlp = None
        self.block.norm2 = None

    def forward(self, x):
        if self.add_global==70:
            x = x + torch.transpose(self.drop_path(torch.transpose(self.block.attn(self.block.norm1(x)), 0, 1)), 0, 1)
        else:
            x = x + self.drop_path(self.block.attn(self.block.norm1(x)))
        # x = x + self.block.drop_path(self.block.mlp(self.block.norm2(x)))
        return x

class MLPDecorder(torch.nn.Module):

    def __init__(self, dim, mlp_dim, skip_mlp=False):
        super().__init__()
        self.skip_mlp = skip_mlp
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, mlp_dim), torch.nn.BatchNorm1d(mlp_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(mlp_dim, dim), torch.nn.BatchNorm1d(dim), )

    def forward(self, x):
        if self.training and not self.skip_mlp :
            bt_x = x[len(x)//2:]
            old_x = x[:len(x)//2]
            bt_x = self.mlp(bt_x)
            x = torch.cat([old_x, bt_x], dim=0)
        return x

class MLPEncoder(torch.nn.Module):

    def __init__(self, d_model, batch_size, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(MLPEncoder, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(batch_size, batch_size)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        import torch.nn.functional as F
        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            import torch.nn.functional as F
            state['activation'] = F.relu
        super(MLPEncoder, self).__setstate__(state)

    def forward(self, src):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = torch.transpose(src, 0, 2)
        src2 = self.linear1(src2)
        src2 = torch.transpose(src2, 0, 2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        return src


class TransformerDecorator1(torch.nn.Module):
    def __init__(self, add_global=3, dim=2048, eval_global=0, dropout=0, small_seq=False, args=None, first_layer=False, drop_path=0.1):
        super(TransformerDecorator1, self).__init__()
        self.not_cls_token = False
        self.no_fp16_bt = 0
        self.shuffle_patch=False
        self.cls_token_only=False
        heads = 4
        self.empty_bt = 0
        self.mlp_enc = 0
        self.batch_size = 128
        self.skip_bt = False
        if args is not None:
            self.not_cls_token = args.not_cls_token
            self.cls_token_only = args.cls_token_only
            self.no_fp16_bt = args.no_fp16_bt
            self.mlp_enc = args.mlp_enc
            heads = args.num_heads
            self.shuffle_patch = args.shuffle_patch
            self.empty_bt = args.empty_bt
            self.batch_size = args.batch_size
            self.all_patches = args.all_patches
            self.skip_bt = args.skip_bt
        self.encoder_layers = torch.nn.TransformerEncoderLayer(dim, heads, dim, 0.5)
        self.eval_global = eval_global
        self.add_global = add_global
        self.first_layer = first_layer
        self.dropout=dropout
        if add_global in [63]:
            self.encoder_layers = TransformerEncoderLayerL(dim, heads, dim, 0.5)
            print('only attention')
        elif add_global in [66]:
            self.encoder_layers = TransformerEncoderLayerA(dim, heads, dim, 0.5)
        elif add_global in [55, 70, 76, 71, 59, 67, 68, 69, 72, 73, 74, 77, 75, 81, 82]:
            # 76 is for 70 (drop patch with mlp) while 74 is for 70 (drop batch)
            # 75 is for 71 (expand batch)
            # 69 is attention only with two branches, 73 is two branches with drop patch
            from functools import partial
            self.encoder_layers = Block(
                dim=dim, num_heads=heads, mlp_ratio=4, qkv_bias=True, drop=args.drop,
                attn_drop=0., drop_path=drop_path, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), act_layer=torch.nn.GELU)
            if add_global in [75, 71]:
                self.encoder_layers = BlockBF(self.encoder_layers)
            if add_global in [70, 74, 77, 69]: # atten only
                self.encoder_layers = AttentionOnly(self.encoder_layers, drop_path=drop_path, add_global=add_global)
        if self.mlp_enc == 1:
            self.encoder_layers = MLPEncoder(dim, args.batch_size)

        if self.empty_bt:
            self.encoder_layers = torch.nn.Identity()
        self.idx = 0
        self.noise = None
        self.layer_num = 1
        self.small_seq = small_seq
        if self.first_layer:
            self.add_global = 53


    def forward(self, feature):
        if self.add_global in [63]:
            return self.encoder_layers(feature)
        if self.training and self.add_global > 0 and not self.skip_bt:
            old_feature = feature
            if self.add_global in [57, 59, 60, 61, 62, 64, 66, 69, 67, 68, 72, 73, 77, ] and not self.first_layer:
                # split
                old_feature = feature[:len(feature)//2]
                if self.add_global not in [61]:
                    feature = feature[-len(feature)//2:]
            if self.shuffle_patch:
                B, L, C = feature.shape
                idx_orig_list = []
                idx_shuffle_list = []
                for i in range(len(feature)):
                    idx = torch.arange(0, L).type(torch.LongTensor)
                    idx_orig = torch.zeros(L).type(torch.LongTensor)
                    idx_shuffle = torch.cat([torch.randperm(1).type(torch.LongTensor),torch.randperm(L -1 ).type(torch.LongTensor)], 0)

                    idx_orig[idx_shuffle] = idx + i*L
                    idx_shuffle_list.append(idx_shuffle+ i * L)
                    # new_features.append(feature[i][idx_shuffle])
                    idx_orig_list.append(idx_orig)
                idx_shuffle = torch.stack(idx_shuffle_list, 0).reshape(-1).to(feature.device)
                idx_orig_list = torch.stack(idx_orig_list, 0).reshape(-1).to(feature.device)
                feature = feature.view(B*L, C)[idx_shuffle].view(B, L, C)
                pass

            size = feature.shape
            if len(size) == 3: #vit
                if self.add_global == 54:
                    feature = feature.view(size[0] * size[1], 1, size[2])
                    feature = self.encoder_layers(feature)
                    feature = feature.view(size)
                elif self.add_global in [64, 65]:
                    # N L C
                    stride = 4
                    feature = torch.transpose(feature, 1, 0) # L N C
                    feature = torch.reshape(feature,(stride * size[1], size[0] // stride, size[2]))
                    feature = self.encoder_layers(feature)
                    feature = torch.reshape(feature,(size[1], size[0], size[2]))
                    feature = torch.transpose(feature, 1, 0)
                else:
                    if self.all_patches:
                        stride = 4
                        # N L C
                        size = feature.shape
                        feature = torch.transpose(feature, 1, 0) # L N C
                        feature = torch.reshape(feature, (stride * size[1], size[0] // stride, size[2]))
                    if isinstance(self.encoder_layers, Block) or \
                            isinstance(self.encoder_layers, Attention) or \
                            isinstance(self.encoder_layers, BlockBF) or isinstance(self.encoder_layers, AttentionOnly):
                        feature = feature.transpose(0, 1)
                    if self.small_seq:
                        feature = torch.reshape(feature,(32, size[0] // 32 * size[1], size[2]))
                        feature = self.encoder_layers(feature)
                        feature = torch.reshape(feature,(size[0], size[1], size[2]))
                    elif self.not_cls_token:
                        feature1 = self.encoder_layers(feature[:, 1:, :])
                        feature = torch.cat([feature[:, :1, :], feature1], dim=1)
                    elif self.cls_token_only:
                        feature1 = self.encoder_layers(feature[:, :1, :])
                        feature = torch.cat([feature1, feature[:, 1:, :]], dim=1)
                    elif self.no_fp16_bt:
                        from torch.cuda.amp import autocast
                        with autocast(enabled=False):
                            feature = self.encoder_layers(feature.float())
                    else:
                        if self.dropout and self.idx > 10:
                            from torch.cuda.amp import autocast
                            with autocast(enabled=False):
                                feature = self.encoder_layers(feature)
                        else:
                            feature = self.encoder_layers(feature)
                    if isinstance(self.encoder_layers, Block) or \
                            isinstance(self.encoder_layers, Attention) or \
                            isinstance(self.encoder_layers, BlockBF) or isinstance(self.encoder_layers, AttentionOnly):
                        feature = feature.transpose(0, 1)
                    if self.all_patches:
                        # recover
                        feature = torch.reshape(feature, (size[1], size[0], size[2]))
                        feature = torch.transpose(feature, 1, 0)
            else:
                feature = feature.view(feature.size(0), feature.size(1), -1)
                feature = torch.transpose(feature, 2, 1)
                feature = self.encoder_layers(feature)
                feature = torch.transpose(feature, 2, 1)
                feature = feature.view(size)
            if self.shuffle_patch:
                feature = feature.view(B*L, C)[idx_orig_list].view(B, L, C)
            if self.add_global in [61]:
                feature = feature[-len(feature)//2:]
            if self.add_global not in [55, 70, 76, 74] and not self.dropout:
                feature = torch.cat([old_feature, feature], dim=0)
                # print(self.noise, self.idx)
            # print(feature.norm())
            return feature
        elif self.add_global and self.eval_global:
            if isinstance(self.encoder_layers, Block) or \
                    isinstance(self.encoder_layers, Attention):
                feature = feature.transpose(0, 1)
            if self.add_global and self.eval_global == 2:
                size = feature.shape
                old_feature = feature
                feature = torch.reshape(feature, (size[0] * size[1], 1, size[2]))
                feature = self.encoder_layers(feature)
                feature = torch.reshape(feature, (size[0], size[1], size[2])) # Acc@1 80.152 Acc@5 95.160 loss 0.849
                # feature = torch.cat([old_feature, feature], dim=0)
            elif self.add_global and self.eval_global == 1:
                feature = self.encoder_layers(feature)
            elif self.add_global and self.eval_global == 3:
                old_feature = feature
                feature = self.encoder_layers(feature)
                # feature = old_feature + feature # 79.628
                feature = torch.cat([old_feature, feature], dim=0)
            if isinstance(self.encoder_layers, Block) or \
                    isinstance(self.encoder_layers, Attention):
                feature = feature.transpose(0, 1)
        return feature


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    if args.add_global:
        encoder_global = TransformerDecorator1(args.add_global, model.num_features, args.eval_global, dropout=args.dropout,
                                            small_seq=args.small_seq, args=args, drop_path=args.drop_path_bt)

        if args.add_global in [53, 63]:
            res_blocks = []
            for i, block in enumerate(model.blocks):
                if args.no_fp16_bt and args.no_fp16_bt not in [4]:
                    res_blocks.append(BlockWrap32(block, args.no_fp16_bt))
                else:
                    res_blocks.append(block)
            model.blocks = torch.nn.Sequential(*res_blocks)
            model.norm = torch.nn.Sequential(model.norm, encoder_global)
        elif args.add_global in [55, 74, 70, 76, 71, 81, 82, 75]:
            res_blocks = []
            if args.decay_drop_bt > 0.:
                step_drop = (args.drop_path_bt - args.decay_drop_bt) / (len(model.blocks) - 2 - args.start_idx)  # do not use start_idx block

                drop_rate_list = [step_drop*(d-args.start_idx -1) + args.decay_drop_bt if d > args.start_idx else 0 for d in range(0, len(model.blocks))]
            else:
                drop_rate_list = [args.drop_path_bt]*(len(model.blocks)+1)
            if args.insert_idx is not None and len(args.insert_idx) > 0:
                insert_list = args.insert_idx
            else:
                insert_list = list(range(args.start_idx, len(model.blocks)))
            if args.add_global in [70, 76, 71, 81, 82, 75]:
                assert args.insert_idx and len(args.insert_idx) > 0, "you should use insert idx"
            for i, block in enumerate(model.blocks):
                if args.no_fp16_bt and args.no_fp16_bt not in [4]:
                    res_blocks.append(BlockWrap32(block, args.no_fp16_bt))
                else:
                    res_blocks.append(block)
                if i in insert_list:
                    if not args.shared_bt:
                        encoder_global = TransformerDecorator1(args.add_global, model.num_features, args.eval_global, dropout=args.dropout,
                                                           small_seq=args.small_seq, args=args, drop_path=drop_rate_list[i])
                    res_blocks.append(encoder_global)
            model.blocks = torch.nn.Sequential(*res_blocks)
            if args.add_norm_bt:
                if not args.shared_bt:
                    encoder_global = TransformerDecorator1(args.add_global, model.num_features, args.eval_global, dropout=args.dropout,
                                                           small_seq=args.small_seq, args=args, drop_path=args.drop_path_bt)
                model.norm = torch.nn.Sequential(model.norm, encoder_global)
        elif args.add_global in [57, 64, 66, 69, 73, 61, 62, 77]:
            # 61, 62 is for half batch
            # model.norm = torch.nn.Sequential(model.norm, encoder_global)
            res_blocks = []
            first_enc = TransformerDecorator1(args.add_global, model.num_features, args.eval_global,
                                              dropout=args.dropout, small_seq=args.small_seq, args=args, first_layer=True, drop_path=args.drop_path_bt)
            old_enc = first_enc.encoder_layers
            del old_enc
            first_enc.encoder_layers = encoder_global.encoder_layers
            if args.no_grad_bt:
                first_enc.encoder_layers.requires_grad_(False)
                encoder_global.encoder_layers.requires_grad_(False)
            nums = 0
            if args.insert_idx is not None and len(args.insert_idx) > 0:
                insert_list = args.insert_idx
            else:
                insert_list = list(range(args.start_idx, len(model.blocks)))
            for i, block in enumerate(model.blocks):
                if args.no_fp16_bt and args.no_fp16_bt not in [4]:
                    res_blocks.append(BlockWrap32(block, args.no_fp16_bt))
                else:
                    res_blocks.append(block)
                if i in insert_list:
                    if insert_list[0] == i: # first layer
                        res_blocks.append(first_enc)
                        first_enc = None
                    else:
                        if not args.shared_bt:
                            encoder_global = TransformerDecorator1(args.add_global, model.num_features, args.eval_global, dropout=args.dropout,
                                                                   small_seq=args.small_seq, args=args, drop_path=args.drop_path_bt)
                        res_blocks.append(encoder_global)
            # model.norm = torch.nn.Identity()
            if args.add_norm_bt:
                if not args.shared_bt:
                    encoder_global = TransformerDecorator1(args.add_global, model.num_features, args.eval_global, dropout=args.dropout,
                                                           small_seq=args.small_seq, args=args, drop_path=args.drop_path_bt)

                model.norm = torch.nn.Sequential(model.norm, encoder_global)
            model.blocks = torch.nn.Sequential(*res_blocks)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "{}_log.txt".format(args.exp_id)), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
