import matplotlib
matplotlib.use('TkAgg')
import nibabel as nib
import numpy as np
import shutil
import os
import imageio
from utils import tran_label
from PIL import Image

def file_unipixel():
    path='D:\zzf\Project\image_segmentation\MT-SWUnet\ACDC_data\labels/train'
    for file in os.listdir(path):
        image=Image.open(os.path.join(path,file))
        array=np.array(image)
        un=np.unique(array)
        if len(un)==3:
            if un[2]==127:
                print(file)
        print(un)
        #print(len(un))

def unipixel():
    image=Image.open('ACDC/labels/test/patient150_frame12_gt/0.png')
    pix=image.getpixel((0,0))
    print(pix)
    array=np.array(image)
    un=np.unique(array)
    print(un)
    print(image.size)

def iamge_data():
    dataset_path = r"D:\zzf\Project\image_segmentation\MT-SWUnet\ACDC\ACDC\database"
    save_path="D:\zzf\Project\image_segmentation\MT-SWUnet\ACDC/images"
    # 原始的train, valid文件夹路径
    train_dataset_path = os.path.join(dataset_path, 'training')
    test_dataset_path = os.path.join(dataset_path, 'testing')
    # 创建train,valid的文件夹
    train_images_path = os.path.join(save_path, 'train')
    test_images_path = os.path.join(save_path, 'test')

    if os.path.exists(train_images_path) == False:
        os.mkdir(train_images_path)
    if os.path.exists(test_images_path) == False:
        os.mkdir(test_images_path)

    # -----------------移动文件夹-------------------------------------------------
    for file_name in os.listdir(train_dataset_path):
        file_path = os.path.join(train_dataset_path, file_name)
        for image in os.listdir(file_path):
            # 查找对应的后缀名，然后保存到文件中
            if len(image.split('.nii.gz')[0]) == 18:
                shutil.copy(os.path.join(file_path, image), os.path.join(train_images_path, image))

    for file_name in os.listdir(test_dataset_path):
        file_path = os.path.join(test_dataset_path, file_name)
        for image in os.listdir(file_path):
            if len(image.split('.nii.gz')[0]) == 18:
                shutil.copy(os.path.join(file_path, image), os.path.join(test_images_path, image))

def label_data():
    dataset_path = r"D:\zzf\Project\image_segmentation\MT-SWUnet\ACDC\ACDC\database"
    save_path="D:\zzf\Project\image_segmentation\MT-SWUnet\ACDC/labels"
    # 原始的train, valid文件夹路径
    train_dataset_path = os.path.join(dataset_path, 'training')
    test_dataset_path = os.path.join(dataset_path, 'testing')
    # 创建train,valid的文件夹
    train_images_path = os.path.join(save_path, 'train')
    test_images_path = os.path.join(save_path, 'test')

    if os.path.exists(train_images_path) == False:
        os.mkdir(train_images_path)
    if os.path.exists(test_images_path) == False:
        os.mkdir(test_images_path)

    # -----------------移动文件夹-------------------------------------------------
    for file_name in os.listdir(train_dataset_path):
        file_path = os.path.join(train_dataset_path, file_name)
        for image in os.listdir(file_path):
            # 查找对应的后缀名，然后保存到文件中
            if len(image.split('.nii.gz')[0]) == 21:
                shutil.copy(os.path.join(file_path, image), os.path.join(train_images_path, image))

    for file_name in os.listdir(test_dataset_path):
        file_path = os.path.join(test_dataset_path, file_name)
        for image in os.listdir(file_path):
            if len(image.split('.nii.gz')[0]) == 21:
                shutil.copy(os.path.join(file_path, image), os.path.join(test_images_path, image))

def re_derct():
    dataset_path = r"D:\zzf\Project\image_segmentation\MT-SWUnet\ACDC_data\labels"
    save_path="D:\zzf\Project\image_segmentation\MT-SWUnet\ACDC_data/labels"
    # 原始的train, valid文件夹路径
    train_dataset_path = os.path.join(dataset_path, 'train')
    test_dataset_path = os.path.join(dataset_path, 'test')
    # 创建train,valid的文件夹
    train_images_path = os.path.join(save_path, 'train')
    test_images_path = os.path.join(save_path, 'test')

    if os.path.exists(train_images_path) == False:
        os.mkdir(train_images_path)
    if os.path.exists(test_images_path) == False:
        os.mkdir(test_images_path)

    # -----------------移动文件夹-------------------------------------------------
    for file_name in os.listdir(train_dataset_path):
        file_path = os.path.join(train_dataset_path, file_name)
        for image in os.listdir(file_path):
            shutil.copy(os.path.join(file_path, image), os.path.join(train_images_path, file_name+"_"+image))

    for file_name in os.listdir(test_dataset_path):
        file_path = os.path.join(test_dataset_path, file_name)
        for image in os.listdir(file_path):
            shutil.copy(os.path.join(file_path, image), os.path.join(test_images_path, file_name+"_"+image))

"""
将多个nii文件（保存在一个文件夹下）转换成png图像。
且图像单个文件夹的名称与nii名字相同。
"""
def nii_to_image(filepath,imgfile):
    filenames = os.listdir(filepath)  # 读取nii文件夹
    slice_trans = []

    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)  # 读取nii
        img_fdata = img.get_fdata()
        fname = f.replace(".nii.gz", "")  # 去掉nii的后缀名
        img_f_path = os.path.join(imgfile, fname)
        # 创建nii对应的图像的文件夹
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)  # 新建文件夹

        # 开始转换为图像
        (x, y, z) = img.shape
        for i in range(z):  # z是图像的序列
            silce = img_fdata[:, :, i]  # 选择哪个方向的切片都可以
            imageio.imwrite(os.path.join(img_f_path, "{}.png".format(i)), silce)
            # 保存图像

def tran_labels():
    path='D:\zzf\Project\image_segmentation\MT-SWUnet\ACDC_data\labels/test'
    for file in os.listdir(path):
        filepath=os.path.join(path,file)
        image=np.array(Image.open(filepath))
        image_map=tran_label(image)

        image_mapped=Image.fromarray(image_map)
        image_mapped.save(filepath)

if __name__ == "__main__":
    # iamge_data()
    # label_data()
    filepath = "ACDC/test"  # nii的文件夹
    imgfile = "ACDC/test"  # image的文件夹
    nii_to_image(filepath,imgfile)

    #re_derct()
    #tran_labels()
    #unipixel()

