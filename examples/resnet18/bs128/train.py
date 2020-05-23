# %% Import packages

import os
import argparse

import numpy as np

from fastai.vision import ImageDataBunch, get_transforms
from pathlib import Path
from torchvision import models

from nccd.io import all_losses_to_array
from nccd.summaries import get_all_losses_per_epoch
from nccd.train import fit_model

# %% Parse command line arguments

model_dict ={
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50
}

parser = argparse.ArgumentParser('Model trainer')

parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--train_dirname', type=str, default='training')
parser.add_argument('--validation_dirname', type=str, default='validation')
parser.add_argument('--test_dirname', type=str, default='test')
parser.add_argument('--ds_tfms', action='store_true')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--model', type=str, choices=list(model_dict.keys()), default='resnet18')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--ps', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--cyc_len', type=int, default=5)
parser.add_argument('--lr_interval', action='store_true')
parser.add_argument('--lr_lower', type=float, default=3e-6)
parser.add_argument('--lr_upper', type=float, default=3e-4)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--unfreeze', action='store_true')
parser.add_argument('--mixup', action='store_true')
parser.add_argument('--output_path', type=str, default=os.getcwd())
parser.add_argument('--output_model_filename', type=str, default='model')
parser.add_argument('--save_loss', action='store_true')
parser.add_argument('--output_training_loss_filename', type=str, default='training_losses.txt')
parser.add_argument('--output_loss_filename', type=str, default='all_losses.csv')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# %% Main function

def main():
    # Set data transformations
    if args.ds_tfms:
        ds_tfms = get_transforms(do_flip=True, flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.1)
    else:
        ds_tfms = None

    # Load data
    data = ImageDataBunch.from_folder(
        args.data_path,
        train=args.train_dirname,
        valid=args.validation_dirname,
        test=args.test_dirname,
        ds_tfms=ds_tfms,
        size=args.image_size,
        bs=args.batch_size
    ).normalize()

    if args.verbose:
        print(
            len(data.train_ds), "training images",
            len(data.valid_ds), "validation images and",
            len(data.test_ds) , "test images"
        )

    # Create output directory if it does not exist
    if not Path(args.output_path).exists():
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # Set up learning rate
    if args.lr_interval:
        max_lr = slice(args.lr_lower, args.lr_upper)
    else:
        max_lr = args.lr

    # Fit model using one-cycle policy
    recorder = fit_model(
        data,
        model_dict[args.model],
        args.ps,
        args.wd,
        args.cyc_len,
        max_lr,
        args.output_path,
        pretrained=args.pretrained,
        unfreeze=args.unfreeze,
        mixup=args.mixup,
        filename=args.output_model_filename,
        verbose=args.verbose
    )

    # Save loss value for each processed batch to file
    if args.save_loss:
        np.savetxt(
            Path(args.output_path).joinpath(args.output_training_loss_filename),
            [l.item() for l in recorder.losses],
            newline='\n',
            header='losses',
            comments=''
        )

        # Save training and validation losses per epoch
        np.savetxt(
            Path(args.output_path).joinpath(args.output_loss_filename),
            all_losses_to_array(get_all_losses_per_epoch(recorder)),
            fmt=['%d', '%f', '%f'],
            delimiter=',',
            newline='\n',
            header='nb_batches,train_losses,val_losses',
            comments=''
        )

    if args.verbose:
        print("Completed execution.")

# %% Execute main function

if __name__ == '__main__':
    main()
