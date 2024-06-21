import argparse
from pathlib import Path
import os
import torch
from models.unet import unet
from utils.data import split_train_val, segmentation_dataset
from torch.utils.data import DataLoader
import torchvision.models.segmentation as tv_seg
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-expname', type=str, required=True, help='experiment name for logging purpose')
    parser.add_argument('-dataset-dir', type=str, required=True, help='directory with the dataset for trn/val')
    parser.add_argument('-backbone', type=str, default='unet', help='the segmentation model to use')

    parser.add_argument('-in-channel', type=int, default=3, help='the number of channels from input image')
    parser.add_argument('-nclass', type=int, default=2, help='the number of output classes')
    parser.add_argument('-seed', type=int, default=0, help='random seed')

    parser.add_argument('-val-ratio', type=float, default=0.1, help='the ratio for train/val split')
    parser.add_argument('-epoch', type=int, default=100, help='maximum training epochs')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-init-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-lr-patience', type=int, default=10, help='epoch patience for learning rate decay')
    parser.add_argument('-lr-decay', type=float, default=0.1, help='decay factor for learning rate')

    parser.add_argument('-gpu-ids', type=int, nargs='+', default=[0, 1, 2, 3], help="choose gpu device. e.g., 0 1 2 3")

    return parser.parse_args()


def get_train_val_imgs(dataset_dir, args):
    train_imgs_list = dataset_dir / f"seed{args.seed}_train_imgs.txt"
    val_imgs_list = dataset_dir / f"seed{args.seed}_val_imgs.txt"
    if not (train_imgs_list.exists() and val_imgs_list.exists()):
        # split train/val images
        all_imgs = (dataset_dir / 'image').glob('*')
        all_imgs = list(image_name.stem for image_name in all_imgs)
        train_imgs, val_imgs = split_train_val(all_imgs, args.val_ratio, args.seed)
        # write train/val splits to txt file
        for split in ['train', 'val']:
            txt_filepath = dataset_dir / f'seed{args.seed}_{split}_imgs.txt'
            with open(txt_filepath, 'w') as f:
                images = locals()[f'{split}_imgs']
                for img in images:
                    f.write(f"{img}\n")
    else:
        # read image names from txt file
        with open(train_imgs_list, 'r') as f:
            train_imgs = f.readlines()
        with open(val_imgs_list, 'r') as f:
            val_imgs = f.readlines()
    return train_imgs, val_imgs


def compute_iou(predict_score, mask):
    predict_label = predict_score > 0.5
    predict_label = predict_label[:, 1, :, :]
    intersection = torch.sum(predict_label * mask, (1, 2))
    union = torch.sum(torch.logical_or(predict_label, mask), (1, 2))
    iou = (intersection + 0.0001) / (union + 0.0001)
    iou = iou.tolist()
    return iou


def compute_pixel_acc(predict_score, mask):
    predict_label = predict_score > 0.5
    predict_label = predict_label[:, 1, :, :]
    mask = mask.squeeze(1)
    pixel_acc = (predict_label==mask).float().mean(dim=[1,2])
    pixel_acc = pixel_acc.tolist()
    return pixel_acc


def main(args):
    # configure computing environment and randomness
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # load train and validation datasets with a list of image names (no file extension)
    dataset_dir = Path(args.dataset_dir)
    train_imgs, val_imgs = get_train_val_imgs(dataset_dir, args)
    train_ds = segmentation_dataset(folder_path=dataset_dir, folder_name=['image', 'mask'], img_list=train_imgs, file_format=['.jpg', '.png'], transform_options=['hflip', 'vflip', 'normalize', 'rotate'], model_phase='train', seed=args.seed)
    val_ds = segmentation_dataset(folder_path=dataset_dir, folder_name=['image', 'mask'], img_list=val_imgs, file_format=['.jpg', '.png'], transform_options=['normalize'], model_phase='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    # define model, loss function, and optimizer
    if args.backbone == 'unet':
        model = unet(img_channel=args.in_channel, mask_channel=args.nclass).to(device)
    elif args.backbone == 'deeplabv3':
        model = tv_seg.deeplabv3_resnet50(num_classes=args.nclass).to(device)
    elif args.backbone == 'fcn':
        model = tv_seg.fcn_resnet50(num_classes=args.nclass).to(device)
    else:
        ValueError('Specified backbone is not supported!')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=args.lr_patience, verbose=True)
    # perform training and validation
    checkpoint_path = Path.cwd() / 'explogs'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    softmax_layer = torch.nn.Softmax(dim=1)
    best_val_iou = 0
    with open(checkpoint_path / f"{args.expname}_train_val_log.txt", 'w') as log_file:
        for epoch_idx in range(args.epoch):
            # train
            model.train()
            for _, (img, mask) in enumerate(train_loader):
                img, mask = img.to(device), mask.to(device)
                optimizer.zero_grad()
                img_predict = model(img) if args.backbone == 'unet' else model(img)['out']
                img_loss = criterion(img_predict, mask)
                img_loss.backward()
                optimizer.step()
            # val
            model.eval()
            val_iou_list = []
            val_PA_list = []
            with torch.no_grad():
                for _, (img, mask) in enumerate(val_loader):
                    img, mask = img.to(device), mask.to(device)
                    img_predict = model(img) if args.backbone == 'unet' else model(img)['out']
                    img_score = softmax_layer(img_predict)
                    val_iou_list = val_iou_list + compute_iou(img_score, mask)
                    val_PA_list = val_PA_list + compute_pixel_acc(img_score, mask)
            val_mean_iou = np.mean(np.asarray(val_iou_list))
            val_mean_PA = np.mean(np.asarray(val_PA_list))
            # log model state with best validation performance
            if val_mean_iou > best_val_iou:
                best_val_iou = val_mean_iou
                model_best_state = {
                    'epoch':        epoch_idx + 1,
                    'state_dict':   model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                    'optimizer':    optimizer.state_dict()
                }
            scheduler.step(val_mean_iou)
            print('Epoch {:04d}/{:04d} with validation IoU: {:.6f} and validation PA: {:.6f}'.format(epoch_idx + 1, args.epoch, val_mean_iou, val_mean_PA))
            log_file.write('Epoch{:04d}/{:04d} with validation IoU: {:.6f} and validation PA: {:.6f}\n'.format(epoch_idx + 1, args.epoch, val_mean_iou, val_mean_PA))
        print('Best validation IoU: {:.6f} at epoch: {:04d}/{:04d}'.format(best_val_iou, model_best_state['epoch'], args.epoch))
        log_file.write('Best validation IoU: {:.6f} at epoch: {:04d}/{:04d}\n'.format(best_val_iou, model_best_state['epoch'], args.epoch))
    torch.save(model_best_state, checkpoint_path / '{}_best_val_epoch_{}.pth'.format(args.expname, model_best_state['epoch']))


if __name__ == '__main__':
    args = get_args()
    main(args)