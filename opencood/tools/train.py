# -*- coding: utf-8 -*-


import argparse
import os
import statistics
import sys
from collections import OrderedDict

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def evaluate_model(model, criterion, data_loader, dataset, device):
    """Evaluate one model pass per test sample and collect detection stats."""
    result_stat = {
        iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
        for iou in IOU_THRESHOLDS
    }
    losses = []
    model.eval()

    with torch.no_grad():
        progress = tqdm.tqdm(
            data_loader,
            desc='Testing',
            leave=False,
            file=train_utils.get_terminal_stream(sys.stderr))
        for batch_data in progress:
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = OrderedDict()
            for cav_id, cav_content in batch_data.items():
                output_dict[cav_id] = model(cav_content)

            ego_output = output_dict['ego']
            loss = criterion(ego_output, batch_data['ego']['label_dict'])
            losses.append(loss.item())

            pred_boxes, pred_scores, gt_boxes = dataset.post_process(
                batch_data, output_dict)
            for iou in IOU_THRESHOLDS:
                eval_utils.caluclate_tp_fp(
                    pred_boxes, pred_scores, gt_boxes, result_stat, iou)

    mean_loss = statistics.mean(losses) if losses else 0.0
    return mean_loss, result_stat


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def build_train_validate_datasets(hypes):
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(
        hypes,
        visualize=False,
        train=False,
        cav_label_only=False)
    return opencood_train_dataset, opencood_validate_dataset


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    if opt.model_dir:
        saved_path = os.path.abspath(opt.model_dir)
    else:
        saved_path = train_utils.setup_train(hypes)
    output_dirs = train_utils.setup_output_dirs(saved_path)
    train_utils.setup_console_log(output_dirs['logs'])

    print('-----------------Dataset Building------------------')
    opencood_train_dataset, opencood_validate_dataset = \
        build_train_validate_datasets(hypes)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=1,
                            num_workers=8,
                            collate_fn=opencood_validate_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(output_dirs['tensorboard'])

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(
            total=len(train_loader),
            leave=True,
            file=train_utils.get_terminal_stream(sys.stderr))

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])


            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(output_dirs['checkpoints'],
                             'net_epoch%d.pth' % (epoch + 1)))

        # Run the complete test split after every training epoch.
        if opt.distributed:
            torch.distributed.barrier()
        is_main_process = not opt.distributed or opt.rank == 0
        if is_main_process:
            valid_ave_loss, result_stat = evaluate_model(
                model_without_ddp,
                criterion,
                val_loader,
                opencood_validate_dataset,
                device)
            display_epoch = epoch + 1
            print('At epoch %d, the test loss is %f' % (
                display_epoch, valid_ave_loss))
            writer.add_scalar('Test/Loss', valid_ave_loss,
                              display_epoch)
            metrics = eval_utils.eval_final_results(
                result_stat, output_dirs['metrics'], epoch=display_epoch)
            for iou, values in metrics.items():
                iou_tag = f'{iou:.1f}'
                writer.add_scalar(f'Test/AP@{iou_tag}', values['ap'],
                                  display_epoch)
                writer.add_scalar(
                    f'Test/Precision@{iou_tag}', values['precision'],
                    display_epoch)
                writer.add_scalar(
                    f'Test/Recall@{iou_tag}', values['recall'],
                    display_epoch)
        if opt.distributed:
            torch.distributed.barrier()

    writer.close()
    print('Training Finished, outputs saved to %s' % saved_path)


if __name__ == '__main__':
    main()
