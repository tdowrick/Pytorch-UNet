
import argparse
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.hausdorff_distance import hausdorff_loss
from utils.data_loading import BasicDataset


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    hausdorff_distance = 0
    
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            # TODO: Test this is ok, i.e. is hausdorff being calculated properly
            hausdorff_distance += hausdorff_loss(mask_pred, mask_true.long())

    net.train()
    return (dice_score + hausdorff_distance) / num_val_batches

def get_args():
    parser = argparse.ArgumentParser(description='Evalute the UNet on images and target masks')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--data', type=str, default=False, help="Data to evaluate")
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    img_scale = args.scale

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    dir_val_img = args.data + '/images'
    dir_val_mask = args.data + '/masks'
    val_dataset = BasicDataset(dir_val_img, dir_val_mask, img_scale)
    
    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=True, drop_last=True, **loader_args)
    val_score = evaluate(net, val_loader, device)

    print(f"{args.data} Mean DICE score: {val_score}")

