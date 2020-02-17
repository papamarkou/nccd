# %% Import packages

import os
import argparse

import numpy as np
import torch

from torchvision import models
from fastai.vision import ImageDataBunch, get_transforms

from nccd.io import get_tile_filename_info
from nccd.io import tile_output_to_array
from nccd.tune import tune_thres
from nccd.predict import predict_tiles

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
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--model', type=str, choices=list(model_type.keys()), default='resnet18')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--thresholds', nargs='+', type=int, default=[1, 2])
parser.add_argument('--tpr_lb', type=int)
parser.add_argument('--fpr_ub', type=int)
parser.add_argument('--output_path', type=str, default=os.getcwd())
parser.add_argument('--output_optimal_thres_filename', type=str, default='optimal_thres.csv')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

print(args)

# %% Main function

def main():
    # Load data
    data = ImageDataBunch.from_folder(
        args.data_path,
        train=args.train_dirname,
        valid=args.test_dirname,
        ds_tfms=get_transforms(do_flip=True, flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.1),
        size=args.image_size,
        bs=args.batch_size
    ).normalize()

    if args.verbose:
        print(len(data.valid_ds), "test images")
        
    optimal_threshold, threshold_f1 = tune_thres(
        data, model_type[args.model], args.model_path, args.thresholds,
        tpr_lb=args.tpr_lb, fpr_ub=args.fpr_ub, verbose=args.verbose
    )

    print(optimal_threshold)

    # Save tile IDs, image IDs, tile prediction scores, tile predictions and tile true labels to file
    #np.savetxt(
    #    os.path.join(args.output_path, args.output_filename),
    #    tile_output,
    #    fmt=['%s', '%s', '%f', '%d', '%d', '%d'],
    #    delimiter=',',
    #    newline='\n',
    #    header='tile_id,image_id,tile_score,tile_pred,tile_target,image_target',
    #    comments=''
    #)

    if args.verbose:
        print("Completed execution.")

# %% Execute main function

if __name__ == '__main__':
    main()
