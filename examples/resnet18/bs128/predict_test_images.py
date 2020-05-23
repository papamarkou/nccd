# %% Import packages

import os
import argparse

import numpy as np

from nccd.io import image_output_to_array
from nccd.predict import predict_images

# %% Parse command line arguments

parser = argparse.ArgumentParser('Predictor')

parser.add_argument('--data_filename', type=str, required=True)
parser.add_argument('--thres', nargs='+', type=int, required=True)
parser.add_argument('--output_path', type=str, default=os.getcwd())
parser.add_argument('--output_filename', type=str, default='image_preds.csv')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# %% Main function

def main():
    # Load data
    data = np.loadtxt(
        args.data_filename,
        dtype={
            'names': ('image_id', 'image_corrosion_count', 'image_target'),
            'formats': ('<U100', int, int)
        },
        delimiter=',',
        skiprows=1
    )

    if args.verbose:
        print(len(data), "test images")
    
    # Convert data from array to dictionary
    image_dict = dict(zip(data['image_id'], [list(v) for v in zip(data['image_corrosion_count'], data['image_target'])]))
    
    # Function for making image predictions using tile predictions
    image_preds = predict_images(image_dict, args.thres)
    
    # Create output directory if it does not exist
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Create numpy array holding the image output
    image_output = image_output_to_array(image_preds)

    # Save image IDs, numer of tiles per image predicted as corroded, image predictions and image true labels to file
    np.savetxt(
        os.path.join(args.output_path, args.output_filename),
        image_output,
        fmt=['%s', '%d', '%d', '%d'],
        delimiter=',',
        newline='\n',
        header='image_id,image_corrosion_count,image_pred,image_target',
        comments=''
    )

    if args.verbose:
        print("Completed execution.")

# %% Execute main function

if __name__ == '__main__':
    main()
