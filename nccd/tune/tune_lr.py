# %% Import packages

from fastai.vision import cnn_learner, accuracy

# %% Function for exploring empirically optimal learning rates

def tune_lr(data, model, pretrained, ps, num_iters):
    # Set up CNN learner
    learner = cnn_learner(data, model, metrics=accuracy, pretrained=pretrained, ps=ps).mixup()

    # Explore possible learning rates
    learner.lr_find(num_it=num_iters)

    # Return learning rates and losses
    return learner.recorder.lrs, learner.recorder.losses
