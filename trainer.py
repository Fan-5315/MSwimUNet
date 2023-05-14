import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss,MIoU
from torchvision import transforms
import cv2
from utils import cal_val

def trainer_synapse(args,model, snapshot_path,worker_init_fn):
    from datasets.dataset_synapse import AcdcDataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    train_data = AcdcDataset(images_dir=os.path.join(args.root_path,'train'), masks_dir=os.path.join(args.list_dir,'train'),
                                   transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    val_data = AcdcDataset(images_dir=os.path.join(args.root_path,'val'), masks_dir=os.path.join(args.list_dir,'val'),
                                 transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(train_data)))
    print("The length of val set is: {}".format(len(val_data)))



    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    miou=MIoU()
    mask_ce_loss = nn.CrossEntropyLoss()       #掩码重构标签任务损失函数

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    iter_num1 = 0
    max_epoch = args.max_epochs
    val_use=args.val_use
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1

    logging.info("{} iterations per epoch. {} max iterati+"
                 "ons ".format(len(train_loader), max_iterations))
    best_performance = 0.0
    best_performance_ = 0.0
    lr_ = base_lr
    device = torch.cuda.get_device_name()
    print('Using {} device'.format(device))
    for epoch_num in range(max_epoch):
        iterator = tqdm(total=len(train_loader), ncols=70)
        running_dice=0
        model.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            iterator.set_description(f"Train | Epoch {epoch_num}")
            image_batch, label_batch = sampled_batch[0]['image'], sampled_batch[0]['label']

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs_seg,outputs_mask,x_mask = model(image_batch)

            loss_ce = ce_loss(outputs_seg, label_batch[:].long())
            loss_dice = dice_loss(outputs_seg, label_batch, softmax=True)
            loss_mask = mask_ce_loss(outputs_mask,label_batch[:].long())

            loss = 0.8*(0.3*loss_ce + 0.7*loss_dice) + 0.2*loss_mask

            running_dice += 1-loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lr_>args.min_lr:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/total_loss', loss, iter_num)
            writer.add_scalar('train/loss_ce', loss_ce, iter_num)
            writer.add_scalar('train/loss_dice', loss_dice, iter_num)
            writer.add_scalar('train/loss_mask', loss_mask, iter_num)

            "迭代write可视化结果"
            if iter_num % 80==0:
                "图像分割"
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs_seg, dim=1), dim=1, keepdim=True)
                outputs=outputs[1,...]*50
                writer.add_image('train/Prediction', outputs, iter_num,dataformats='CHW')
                labs = label_batch[1, ...].unsqueeze(0)*50
                writer.add_image('train/GroundTruth', labs, iter_num,dataformats='CHW')
                '掩码重构'
                mask_img = x_mask[1, 0:1, :, :]
                mask = (mask_img - mask_img.min()) / (mask_img.max() - mask_img.min())
                writer.add_image('train/MaskImage', mask, iter_num)
                output_mask = torch.argmax(torch.softmax(outputs_mask, dim=1), dim=1, keepdim=True)
                output_mask = output_mask[1,...]*50
                writer.add_image('train/MaskPrediction', output_mask, iter_num)
            logging.info('iteration %d : total_loss : %f, loss_ce: %f,loss_dice: %f,loss_mask: %f' % (iter_num, loss.item(), loss_ce.item(),loss_dice.item(),loss_mask.item()))

            iterator.set_postfix()
            iterator.update(1)
        iterator.close()

        writer.add_scalar('train/dice', running_dice/len(train_loader), epoch_num)

        "模型保存"
        save_interval = 50
        if epoch_num > int(max_epoch / 4) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
        if running_dice/len(train_loader) > best_performance_:   #一轮epoch的最大dice
            best_performance_ = running_dice/len(train_loader)
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num>0:
            if val_use:
                with torch.no_grad():
                    model.eval()
                    iterator1 = tqdm(total=len(val_loader), ncols=70)
                    epoch_dice=0
                    n_batch=0
                    running_performance=0
                    metric_list = 0.0
                    for i_batch, sampled_batch in enumerate(val_loader):
                        iterator1.set_description(f"Val | Epoch {epoch_num}")
                        image_batch, label_batch = sampled_batch[0]['image'], sampled_batch[0]['label']
                        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                        outputs_seg, outputs_mask, x_mask = model(image_batch)

                        loss_ce = ce_loss(outputs_seg, label_batch[:].long())
                        loss_dice = dice_loss(outputs_seg, label_batch, softmax=True)
                        loss_mask = mask_ce_loss(outputs_mask, label_batch[:].long())

                        "miou计算"
                        mIoU_num=miou.compute_mIoU(outputs_seg,label_batch,num_classes)
                        "total loss"
                        loss = 0.8 * (0.4 * loss_ce + 0.6 * loss_dice) + 0.2 * loss_mask
                        "Dice系数"
                        metric_i = cal_val(image_batch, label_batch, model, classes=args.num_classes)
                        metric_list += np.array(metric_i)
                        Dice=np.mean(metric_i, axis=0)[0]

                        running_performance += Dice
                        iter_num1 = iter_num1 + 1
                        n_batch += 1
                        writer.add_scalar('val/total_mean_loss', loss, iter_num1)
                        writer.add_scalar('val/MIoU', mIoU_num, iter_num1)

                        if iter_num1 % 200 == 0:
                            "图像分割"
                            image = image_batch[0, 0:1, :, :]
                            image = (image - image.min()) / (image.max() - image.min())
                            writer.add_image('val/Image', image, iter_num1)
                            outputs = torch.argmax(torch.softmax(outputs_seg, dim=1), dim=1, keepdim=True)
                            outputs = outputs[0, ...] * 50
                            writer.add_image('val/Prediction', outputs, iter_num1, dataformats='CHW')
                            labs = label_batch[0, ...].unsqueeze(0) * 50
                            writer.add_image('val/GroundTruth', labs, iter_num1, dataformats='CHW')
                            '掩码重构'
                            mask_img = x_mask[0, 0:1, :, :]
                            mask = (mask_img - mask_img.min()) / (mask_img.max() - mask_img.min())
                            writer.add_image('val/MaskImage', mask, iter_num1)
                            output_mask = torch.argmax(torch.softmax(outputs_mask, dim=1), dim=1, keepdim=True)
                            output_mask = output_mask[0, ...] * 50
                            writer.add_image('val/MaskPrediction', output_mask, iter_num1)
                        logging.info('iteration %d : total_loss : %f, MIoU: %f' % (iter_num1, loss.item(),mIoU_num))

                        iterator1.set_postfix()
                        iterator1.update(1)
                iterator1.close()
                epoch_dice = running_performance / len(val_loader)
                writer.add_scalar('val/Dice', epoch_dice, epoch_num)
                logging.info('Dice: %f' % epoch_dice)

                if epoch_dice > best_performance:
                    best_performance = epoch_dice
                    save_mode_path = os.path.join(snapshot_path, 'val_best.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"