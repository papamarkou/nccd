# %% Import packages

import os
import argparse

import numpy as np

from nccd.io import image_corrosion_output_to_array
from nccd.summarize import count_image_corrosion

# %% Parse command line arguments

parser = argparse.ArgumentParser('Predictor')

parser.add_argument('--data_filename', type=str, required=True)
parser.add_argument('--output_path', type=str, default=os.getcwd())
parser.add_argument('--output_filename', type=str, default='image_corrosion_counts.csv')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# %% Temporary workspace

data = np.loadtxt(
    "validation_tile_preds.csv",
    dtype={
        'names': ('tile_id', 'image_id', 'tile_score', 'tile_pred', 'tile_target', 'image_target'),
        'formats': ('<U100', 'U100', float, int, int, int)
    },
    delimiter=',',
    skiprows=1
)

image_dict = count_image_corrosion(data['image_id'], data['tile_pred'], data['tile_target'], data['image_target'])

# %% Main function

def main():
    # Load data
    data = np.loadtxt(
        args.data_filename,
        dtype={
            'names': ('tile_id', 'image_id', 'tile_score', 'tile_pred', 'tile_target', 'image_target'),
            'formats': ('<U100', 'U100', float, int, int, int)
        },
        delimiter=',',
        skiprows=1
    )

    if args.verbose:
        print(len(data), "test images")

    # Count number of corroded tiles per image
    image_dict = count_image_corrosion(data['image_id'], data['tile_pred'], data['tile_target'], data['image_target'])


    # Create output directory if it does not exist
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Create numpy array holding the image output
    image_output = image_corrosion_output_to_array(image_dict)

    # Save image IDs, numer of tiles per image predicted as corroded and image true labels to file
    np.savetxt(
        os.path.join(args.output_path, args.output_filename),
        image_output,
        fmt=['%s', '%d', '%d'],
        delimiter=',',
        newline='\n',
        header='image_id,image_corrosion_count,image_target',
        comments=''
    )

    if args.verbose:
        print("Completed execution.")

# %% Execute main function

if __name__ == '__main__':
    main()
