# %% Import packages

import numpy as np

from fastai.vision import DatasetType, cnn_learner, accuracy

# %% Function for computing tile prediction scores using trained model

def predict_tiles(data, model_type, trained_model):
    # Set up CNN learner
    learner = cnn_learner(data, model_type, metrics=accuracy).mixup()

    # Load trained model
    learner.load(trained_model)

    # Compute precition score
    tile_scores, tile_targets = learner.get_preds(ds_type=DatasetType.Valid)

    # Return predictions and true labels
    return tile_scores, tile_targets

# %% Function for placing tile output to a numpy array
    
def tile_output_to_array(tile_ids, image_ids, tile_scores, tile_preds, tile_targets, image_targets):
    tile_output = np.empty(
        len(tile_ids),
        dtype=[
            ('col1', '<U{0}'.format(np.max([len(id) for id in tile_ids]))),
            ('col2', '<U{0}'.format(np.max([len(id) for id in image_ids]))),
            ('col3', float),
            ('col4', int),
            ('col5', int),
            ('col6', int)
        ]
    )
    
    tile_output['col1'] = tile_ids
    tile_output['col2'] = image_ids
    tile_output['col3'] = tile_scores[: , 1].clone().detach().numpy()
    tile_output['col4'] = tile_preds.clone().detach().cpu()
    tile_output['col5'] = tile_targets.clone().detach().cpu()
    tile_output['col6'] = image_targets
    
    return tile_output
