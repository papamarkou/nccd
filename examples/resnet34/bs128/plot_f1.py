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
parser.add_argument('--output_filename', type=str, default='f1_plot.jpg')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# %% Main function

def main():
    # Load data
    data = np.loadtxt(
        args.data_filename,
        dtype={'names': ('thres', 'f1' ,'tpr' , 'fpr'), 'formats': (float, float, float, float)},
        delimiter=',',
        skiprows=1
    )

    # Plot learning rate against loss
    plt.figure()
    # plt.rc('text', usetex=True)
    with plt.rc_context({'lines.linewidth': 2.5}):
        sns.lineplot(data['thres'], data['f1'], marker='o')
    plt.scatter(21, 1, marker='o', s=200, color='red')
    plt.xticks(ticks=[1, 5, 10, 15, 20, 25, 30], labels=['1', '5', '10', '15', '20', '25', '30'])
    plt.xlabel('Threshold c (hyperparameter)')
    plt.ylabel('F1 score')

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
