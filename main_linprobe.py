# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
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

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

import models_vit

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

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
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
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
    parser.add_argument('--put_decoder', action='store_true',)
    
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

    # linear probe: weak augmentation
    transform_train = transforms.Compose([
            RandomResizedCrop(224, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
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

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    )
    if args.add_global and not args.put_decoder:
        from bt import TransformerDecorator1, BlockWrap, BlockBF, BlockWrap32, BlockWrapDebug, AttentionOnly, MLPDecorder, MLPEncoder
        
        num_features = model.num_features
        encoder_global = TransformerDecorator1(args.add_global, num_features, args.eval_global, dropout=args.dropout,
                                            small_seq=args.small_seq, args=args, drop_path=args.drop_path_bt)
        insert_blocks = model.blocks
        
        if args.put_decoder:
            insert_blocks = model.decoder_blocks
        if args.add_global in [53, 63]:
            res_blocks = []
            for i, block in enumerate(insert_blocks):
                if args.no_fp16_bt and args.no_fp16_bt not in [4]:
                    res_blocks.append(BlockWrap32(block, args.no_fp16_bt))
                else:
                    res_blocks.append(block)
            model.norm = torch.nn.Sequential(model.norm, encoder_global)
        elif args.add_global in [55, 74, 70, 76, 71, 81, 82, 75]:
            res_blocks = []
            if args.decay_drop_bt > 0.:
                step_drop = (args.drop_path_bt - args.decay_drop_bt) / (len(insert_blocks) - 2 - args.start_idx)  # do not use start_idx block

                drop_rate_list = [step_drop*(d-args.start_idx -1) + args.decay_drop_bt if d > args.start_idx else 0 for d in range(0, len(insert_blocks))]
            else:
                drop_rate_list = [args.drop_path_bt]*(len(insert_blocks)+1)
            if args.insert_idx is not None and len(args.insert_idx) > 0:
                insert_list = args.insert_idx
            else:
                insert_list = list(range(args.start_idx, len(insert_blocks)))
            if args.add_global in [70, 76, 71, 81, 82, 75]:
                assert args.insert_idx and len(args.insert_idx) > 0, "you should use insert idx"
            for i, block in enumerate(insert_blocks):
                if args.no_fp16_bt and args.no_fp16_bt not in [4]:
                    res_blocks.append(BlockWrap32(block, args.no_fp16_bt))
                else:
                    res_blocks.append(block)
                if i in insert_list:
                    if not args.shared_bt:
                        encoder_global = TransformerDecorator1(args.add_global, num_features, args.eval_global, dropout=args.dropout,
                                                           small_seq=args.small_seq, args=args, drop_path=drop_rate_list[i])
                    res_blocks.append(encoder_global)
            if args.add_norm_bt:
                if not args.shared_bt:
                    encoder_global = TransformerDecorator1(args.add_global, num_features, args.eval_global, dropout=args.dropout,
                                                           small_seq=args.small_seq, args=args, drop_path=args.drop_path_bt)
                model.norm = torch.nn.Sequential(model.norm, encoder_global)
        elif args.add_global in [57, 64, 66, 69, 73, 61, 62, 77]:
            # 61, 62 is for half batch
            # model.norm = torch.nn.Sequential(model.norm, encoder_global)
            res_blocks = []
            first_enc = TransformerDecorator1(args.add_global, num_features, args.eval_global,
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
                insert_list = list(range(args.start_idx, len(insert_blocks)))
            for i, block in enumerate(insert_blocks):
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
                            encoder_global = TransformerDecorator1(args.add_global, num_features, args.eval_global, dropout=args.dropout,
                                                                   small_seq=args.small_seq, args=args, drop_path=args.drop_path_bt)
                        res_blocks.append(encoder_global)
            # model.norm = torch.nn.Identity()
            if args.add_norm_bt:
                if not args.shared_bt:
                    encoder_global = TransformerDecorator1(args.add_global, num_features, args.eval_global, dropout=args.dropout,
                                                           small_seq=args.small_seq, args=args, drop_path=args.drop_path_bt)

                model.norm = torch.nn.Sequential(model.norm, encoder_global)
        else:
            res_blocks = insert_blocks
            
        if args.put_decoder:
            model.decoder_blocks = torch.nn.Sequential(*res_blocks)
        else:
            model.blocks = torch.nn.Sequential(*res_blocks)
        print(model)
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
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
