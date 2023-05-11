import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,default='./ACDC_data/images', help='train root dir for data')  #训练集样本路径
parser.add_argument('--dataset', type=str,default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,default='./ACDC_data/labels', help='list dir')   #训练集样本标签路径
parser.add_argument('--num_classes', type=int,default=4, help='output channel of network')
parser.add_argument('--mask_rate', type=float,default=0.15, help='')
parser.add_argument('--mask_size', type=float,default=4, help='')
parser.add_argument('--val_use', default=True, help='whether use val data')
parser.add_argument('--output_dir', type=str, default='./output',help='output dir')
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,default=600, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,default=1,help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.03,help='segmentation network learning rate')
parser.add_argument('--min_lr', type=float,  default=0.00005,help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml',required=False, metavar="FILE", help='path to config file')
parser.add_argument("--opts",default=None,nargs='+',help="Modify config options by adding 'KEY VALUE' pairs.")
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],help='no: no cache, ''full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=False,help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, default=False,help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(config, img_size=args.img_size, mask_rate=args.mask_rate,mask_size=args.mask_size,num_classes=args.num_classes).cuda()
    net.load_from(config)

    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args,net, args.output_dir,worker_init_fn)