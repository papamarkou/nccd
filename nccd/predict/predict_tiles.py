# %% Import packages

from fastai.vision import DatasetType, cnn_learner, accuracy

# %% Function for computing tile prediction scores using trained model

def predict_tiles(data, model_type, trained_model, ps, wd, mixup=True):
    # Set up CNN learner
    learner = cnn_learner(data, model_type, metrics=accuracy, ps=ps, wd=wd)
    if mixup:
        learner = learner.mixup()

    # Load trained model
    learner.load(trained_model)

    # Compute precition score
    tile_scores, tile_targets = learner.get_preds(ds_type=DatasetType.Valid)

    # Return predictions and true labels
    return tile_scores, tile_targets
