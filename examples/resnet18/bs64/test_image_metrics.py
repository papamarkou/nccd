# %% Import packages

import os
import argparse

import numpy as np

from nccd.io import image_metrics_to_array
from nccd.summaries import error_metrics

# %% Parse command line arguments

parser = argparse.ArgumentParser('Predictor')

parser.add_argument('--data_filename', type=str, required=True)
parser.add_argument('--output_path', type=str, default=os.getcwd())
parser.add_argument('--output_filename', type=str, default='image_metrics.csv')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# %% Main function

def main():
    # Load data
    data = np.loadtxt(
        args.data_filename,
        dtype={
            'names': ('image_id', 'image_corrosion_count', 'image_pred', 'image_target'),
            'formats': ('<U100', int, int, int)
        },
        delimiter=',',
        skiprows=1
    )

    if args.verbose:
        print(len(data), "test images")
    
    # Compute error metrics for images
    image_metrics = error_metrics(data['image_target'], data['image_pred'])

    # Save image metrics to file
    np.savetxt(
        os.path.join(args.output_path, args.output_filename),
        image_metrics_to_array(image_metrics),
        fmt=['%s', '%f'],
        delimiter=',',
        newline='\n',
        header='metric_name,metric_value',
        comments=''
    )

    if args.verbose:
        print("Completed execution.")

# %% Execute main function

if __name__ == '__main__':
    main()
