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
        dtype={'names': ('nb_batches', 'train_losses', 'val_losses'), 'formats': (int, float, float)},
        delimiter=',',
        skiprows=1
    )
    
    num_epochs = len(data)

    # Plot learning rate against loss
    plt.figure()
    # plt.rc('text', usetex=True)
    with plt.rc_context({'lines.linewidth': 2.5}):
        sns.lineplot(list(range(1, num_epochs+1)), data['train_losses'], marker='o', label="Training loss")
        sns.lineplot(list(range(1, num_epochs+1)), data['val_losses'], marker='o', label="Validation loss")
    plt.ylim([
        0.95*min(min(data['train_losses']), min(data['val_losses'])),
        1.05*max(max(data['train_losses']), max(data['val_losses']))
    ])
    plt.xticks(ticks=list(range(1, num_epochs+1)), labels=[str(epoch) for epoch in list(range(1, num_epochs+1))])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

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
