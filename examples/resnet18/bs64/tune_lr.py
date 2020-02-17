# %% Import packages

import os
import argparse

import numpy as np

from torchvision import models
from fastai.vision import ImageDataBunch, get_transforms

from nccd.tune import tune_lr

# %% Parse command line arguments

model_dict ={
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50
}

parser = argparse.ArgumentParser('Model tuner')

parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--train_dirname', type=str, default='training')
parser.add_argument('--validation_dirname', type=str, default='validation')
parser.add_argument('--test_dirname', type=str, default='test')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--model', type=str, choices=list(model_dict.keys()), default='resnet18')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--ps', type=float, default=0.1)
parser.add_argument('--start_lr', type=float, default=0.0000001)
parser.add_argument('--end_lr', type=float, default=1.)
parser.add_argument('--num_lr_iters', type=int, default=100)
parser.add_argument('--output_path', type=str, default=os.getcwd())
parser.add_argument('--output_filename', type=str, default='lrs_and_losses.csv')
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

    # Explore optimal learning rates
    lrs, losses = tune_lr(
        data, model_dict[args.model], args.pretrained, args.ps, args.start_lr, args.end_lr, args.num_lr_iters
    )

    # Create output directory if it does not exist
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Save learning rates and associated loss values
    np.savetxt(
        os.path.join(args.output_path, args.output_filename),
        np.column_stack((lrs, losses)),
        delimiter=',',
        newline='\n',
        header='lrs,losses',
        comments=''
    )

    if args.verbose:
        print("Completed execution.")

# %% Execute main function

if __name__ == '__main__':
    main()
