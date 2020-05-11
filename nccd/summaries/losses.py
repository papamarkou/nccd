# %% Import packages

import itertools

# %%

def get_train_losses_per_epoch(losses, nb_batches):
    train_losses_per_epoch = []

    for c in itertools.accumulate(nb_batches):
        train_losses_per_epoch.append(losses[c-1].item())

    return train_losses_per_epoch

# %%

def get_all_losses_per_epoch(recorder):
    losses = dict(
        nb_batches=[c for c in itertools.accumulate(recorder.nb_batches)],
        train_losses=get_train_losses_per_epoch(recorder.losses, recorder.nb_batches),
        val_losses=recorder.val_losses
    )

    return losses
