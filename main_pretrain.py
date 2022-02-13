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
        from bt import TransformerDecorator1, BlockWrap, BlockBF, BlockWrap32, BlockWrapDebug, AttentionOnly, MLPDecorder, MLPEncoder
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
