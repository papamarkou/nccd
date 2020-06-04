# %% Import packages

import os
import argparse

import numpy as np

from torchvision import models
from fastai.vision import ImageDataBunch, get_transforms

from nccd.tune import tune_thres

# %% Parse command line arguments

model_type ={
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50
}

parser = argparse.ArgumentParser('Predictor')

parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--train_dirname', type=str, default='training')
parser.add_argument('--test_dirname', type=str, default='test')
parser.add_argument('--ds_tfms', action='store_true')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--model', type=str, choices=list(model_type.keys()), default='resnet18')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--ps', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--thres', nargs='+', type=int, default=[1, 2])
parser.add_argument('--mixup', action='store_true')
parser.add_argument('--tpr_lb', type=int)
parser.add_argument('--fpr_ub', type=int)
parser.add_argument('--output_path', type=str, default=os.getcwd())
parser.add_argument('--output_optimal_thres_filename', type=str, default='optimal_thres.csv')
parser.add_argument('--output_thres_metrics_filename', type=str, default='thres_metrics.csv')
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
        valid=args.test_dirname,
        ds_tfms=ds_tfms,
        size=args.image_size,
        bs=args.batch_size
    ).normalize()

    if args.verbose:
        print(len(data.valid_ds), "test images")

    # Compute F1, TPR and FPR for different thresholds and find optimal threshold
    optimal_thres, optimal_thres_idx, thres_metrics = tune_thres(
        data, model_type[args.model], args.model_path, args.ps, args.wd, args.thres,
        mixup=args.mixup, tpr_lb=args.tpr_lb, fpr_ub=args.fpr_ub, verbose=args.verbose
    )

    # Save optimal threshold and associated F1 score, TPF and FPR to file
    np.savetxt(
        os.path.join(args.output_path, args.output_optimal_thres_filename),
        [[
            optimal_thres,
            thres_metrics['f1'][optimal_thres_idx],
            thres_metrics['tpr'][optimal_thres_idx],
            thres_metrics['fpr'][optimal_thres_idx]
        ]],
        fmt=['%f', '%f', '%f', '%f'],
        delimiter=',',
        newline='\n',
        header='thres,f1,tpr,fpr',
        comments=''
    )

    # Save thresholds, F1 score, TPR and FPR to file
    np.savetxt(
        os.path.join(args.output_path, args.output_thres_metrics_filename),
        thres_metrics,
        fmt=['%f', '%f', '%f', '%f'],
        delimiter=',',
        newline='\n',
        header='thres,f1,tpr,fpr',
        comments=''
    )

    if args.verbose:
        print("Completed execution.")

# %% Execute main function

if __name__ == '__main__':
    main()
