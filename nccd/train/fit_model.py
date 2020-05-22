# %% Import packages

import os

from fastai.vision import cnn_learner, accuracy, callbacks

# %% Function for fitting model using one-cycle policy

def fit_model(
    data, model, ps, wd, cyc_len, max_lr, path,
    pretrained=True, unfreeze=False, mixup=True, filename=None, verbose=True
):
    # Set up CNN learner
    learner = cnn_learner(data, model, metrics=accuracy, pretrained=pretrained, ps=ps, wd=wd)
    if mixup:
        learner = learner.mixup()

    if pretrained and unfreeze:
        learner.unfreeze()

    # Fit model using one-cycle policy
    learner.fit_one_cycle(
        cyc_len,
        max_lr=max_lr,
        callbacks=(callbacks.SaveModelCallback(learner, every='improvement', monitor='accuracy', name='best'))
    )

    # Save trained model
    output_path = learner.save(os.path.join(path, filename), return_path=True)

    if verbose:
        print("Model saved at", output_path)

    return learner.recorder
