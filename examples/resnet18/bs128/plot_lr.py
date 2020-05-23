# %% Import packages

import os
import argparse

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %% Parse command line arguments

parser = argparse.ArgumentParser('Model tuner')

parser.add_argument('--data_filename', type=str, required=True)
parser.add_argument('--output_path', type=str, default=os.getcwd())
parser.add_argument('--output_filename', type=str, default='lr_plot.jpg')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# %% Main function

def main():
    # Load data
    data = np.loadtxt(
        args.data_filename,
        dtype={'names': ('lrs', 'losses'), 'formats': (float, float)},
        delimiter=',',
        skiprows=1
    )

    # Plot learning rate against loss
    plt.figure()
    # plt.rc('text', usetex=True)
    sns.lineplot(np.log10(data['lrs']), data['losses'])
    plt.xticks(
        ticks=range(-7, 1),
        labels=[
            r'$10^{-7}$',
            r'$10^{-6}$',
            r'$10^{-5}$',
            r'$10^{-4}$',
            r'$10^{-3}$',
            r'$10^{-2}$',
            r'$10^{-1}$',
            r'$10^{0}$'
        ]
    )
    plt.xlabel('Learning rate')
    plt.ylabel('Loss')

    # Create output directory if it does not exist
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Save plot to file
    plt.savefig(
        os.path.join(args.output_path, args.output_filename),
        quality=100,
        transparent=True,
        bbox_inches='tight',
    )

    if args.verbose:
        print("Completed execution.")

# %% Execute main function

if __name__ == '__main__':
    main()
