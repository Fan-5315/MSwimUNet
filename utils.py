import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import os
from PIL import Image
import cv2
import matplotlib.colors as mcolors

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class MIoU():
    def compute_iou(self,pred, label, num_classes):
        ious = []
        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = label == cls
            intersection = (pred_inds[target_inds]).long().sum().item()
            union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
            if union == 0:
                ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / float(max(union, 1)))
        return ious

    def compute_mIoU(self,pred_batch, label_batch, num_classes):
        """
        args：
        pred_batch:模型输入的预测张量（batch num_class h w）
        label_batch:实际标签数据（batch 1 h w）
        num_class:分类数
        """
        ious = []
        for i in range(len(pred_batch)):
            pred = torch.argmax(pred_batch[i], dim=0)
            ious.append(self.compute_iou(pred, label_batch[i], num_classes))
        ious = torch.tensor(ious, dtype=torch.float32)
        miou = torch.mean(ious[~torch.isnan(ious)])
        return miou.item()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def dice_calculate(score, target):
    smooth = 1e-5
    intersect = np.sum(score * target)
    y_sum = np.sum(target * target)
    z_sum = np.sum(score * score)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return dice

def one_hot_encoder(input,classes):
    array_list = []
    for i in range(classes):
        temp_prob = (input == i).astype(np.float32)  # * torch.ones_like(input_tensor)
        temp_prob = np.expand_dims(temp_prob,axis=0)
        array_list.append(temp_prob)
    output_array = np.concatenate(array_list, axis=0)
    return output_array

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image= image.squeeze(0).cpu().detach().numpy()
    case=case[0]
    if len(image.shape) == 3:
        input = torch.from_numpy(image).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs,outputs_mask,x_mask= net(input)
            "miou计算"
            miou = MIoU()
            mIoU_num = miou.compute_mIoU(outputs, label, classes)
            x_mask = x_mask.squeeze(0).cpu().detach().numpy()
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out1 = torch.argmax(torch.softmax(outputs_mask, dim=1), dim=1).squeeze(0)
            out_cal = out.cpu().detach().numpy()
            out1_cal = out1.cpu().detach().numpy()
            prediction = out_cal
            prediction1 = out1_cal

    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    label = label.squeeze(0).cpu().detach().numpy()
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    "贴图"
    overImg = overlay_labels(image, prediction)
    overImg1 = overlay_labels(x_mask, prediction1)

    if test_save_path is not None:
        image = np.transpose(image.astype(np.uint8), (1, 2, 0))
        x_mask = np.transpose(x_mask.astype(np.uint8), (1, 2, 0))
        prediction = prediction.astype(np.uint8)
        prediction1 = prediction1.astype(np.uint8)
        label = label.astype(np.uint8)

        image_path = test_save_path + '/' + case[:-4] + '_' + 'image' + '.png'
        overImg_path = test_save_path + '/' + case[:-4] + '_' + 'Task1Img' + '.png'
        overImg1_path = test_save_path + '/' + case[:-4] + '_' + 'Task2Img' + '.png'
        x_mask_path = test_save_path + '/' + case[:-4] + '_' + 'mask_image' + '.png'
        prediction_path = test_save_path + '/' + case[:-4] + '_' + 'pred' + '.png'
        label_path = test_save_path + '/' + case[:-4] + '_' + 'label' + '.png'
        prediction1_path = test_save_path + '/' + case[:-4] + '_' + 'mask_pred' + '.png'

        cv2.imwrite(image_path,image)
        cv2.imwrite(overImg_path, overImg)
        cv2.imwrite(overImg1_path, overImg1)
        cv2.imwrite(x_mask_path, x_mask)
        cv2.imwrite(prediction_path,prediction*50)
        cv2.imwrite(label_path,label*50)
        cv2.imwrite(prediction1_path, prediction1 * 50)
    return metric_list,mIoU_num

def cal_val(image, label, net, classes):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        outputs = net(input)[0]
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        out_cal = out.cpu().detach().numpy()
    prediction = out_cal
    metric_list = []

    for i in range(0, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list

def get_patch_mask(array, mask_rate, patch_size):
    batch_size, channels, height, width = array.shape

    # 计算掩码patch的数量
    n_patches_h = height // patch_size
    n_patches_w = width // patch_size

    # 生成一个 (batch_size, channels, n_patches_h, n_patches_w) 的一维掩码张量，每个元素的值为0或1
    mask_2d = torch.zeros(batch_size,1,n_patches_h,n_patches_w).bernoulli_(1 - mask_rate)

    # 将掩码张量的尺寸缩小到 (num, num)，然后复制到每个图像的相应位置
    mask_2d_patches = torch.nn.functional.interpolate(mask_2d, scale_factor=patch_size, mode='nearest')

    # 输出三维掩码张量
    mask_patches = torch.cat([mask_2d_patches,mask_2d_patches,mask_2d_patches], dim=1)

    return mask_patches.float().to("cuda")


def tran_label(label_batch):
    mapping=np.array([0,1,2,3])

    label_batch_mapped=np.where(label_batch == 0,mapping[0],label_batch)
    label_batch_mapped = np.where(label_batch == 85, mapping[1], label_batch_mapped)
    label_batch_mapped = np.where(label_batch == 170, mapping[2], label_batch_mapped)
    label_batch_mapped = np.where(label_batch == 255, mapping[3], label_batch_mapped)

    return label_batch_mapped


def overlay_labels(original_image, label_image):
    original_image = np.transpose(original_image.astype(np.uint8), (1, 2, 0))
    # 定义颜色映射（标签值与颜色的对应关系）bgra色彩
    label_colors = {
        0: (0, 0, 0, 0),  # 黑色：背景
        1: (150, 255, 255, 255),  # 黄色：Rv（右心室腔）
        2: (203, 230, 0, 128),  # 青色：Myo（心肌）
        3: (213, 192, 255, 128)  # 粉色：Lv（左心室腔）
    }

    # 创建带有透明度通道的RGBA图像
    rgba_image = np.zeros((label_image.shape[0], label_image.shape[1], 4), dtype=np.uint8)
    for label_value, color in label_colors.items():
        mask = label_image == label_value
        rgba_image[mask] = color

    # 将标签图像贴图到原始图像上
    result_image = original_image.copy()
    result_image[rgba_image[..., 3] > 0] = rgba_image[rgba_image[..., 3] > 0, :3]

    # 返回贴图后的图像
    return result_image