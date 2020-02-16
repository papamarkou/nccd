# %% Import packages

import os
import argparse

import numpy as np

from torchvision import models
from fastai.vision import ImageDataBunch, get_transforms

from nccd import fit_model

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
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--model', type=str, choices=list(model_dict.keys()), default='resnet18')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--ps', type=float, default=0.2)
parser.add_argument('--cyc_len', type=int, default=5)
parser.add_argument('--lr_lower', type=float, default=1e-4)
parser.add_argument('--lr_upper', type=float, default=1e-1)
parser.add_argument('--output_path', type=str, default=os.getcwd())
parser.add_argument('--output_model_filename', type=str, default='model')
parser.add_argument('--save_loss', action='store_true')
parser.add_argument('--output_loss_filename', type=str, default='losses.txt')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# %% Main function

def main():
    # Load data
    data = ImageDataBunch.from_folder(
        args.data_path,
        train=args.train_dirname,
        valid=args.validation_dirname,
        test=args.test_dirname,
        ds_tfms=get_transforms(do_flip=True, flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.1),
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
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Fit model using one-cycle policy
    losses = fit_model(
        data,
        model_dict[args.model],
        args.pretrained,
        args.ps,
        args.cyc_len,
        args.lr_lower,
        args.lr_upper,
        args.output_path,
        args.output_model_filename,
        args.verbose
    )

    # Save loss value for each processed batch
    if args.save_loss:
        np.savetxt(
            os.path.join(args.output_path, args.output_loss_filename),
            [l.item() for l in losses],
            newline='\n',
            header='losses',
            comments=''
        )

    if args.verbose:
        print("Completed execution.")

# %% Execute main function

if __name__ == '__main__':
    main()
