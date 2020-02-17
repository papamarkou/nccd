# %% Import packages

import numpy as np

import torch

from fastai.vision import DatasetType, cnn_learner, accuracy

from nccd.io import get_tile_filename_info, image_output_to_array
from nccd.summaries import count_image_corrosion, error_metrics
from nccd.tune import predict_images

# %% Function for computing tile prediction scores using trained model

def tune_thres(data, model_type, trained_model, thresholds, tpr_lb=None, fpr_ub=None, verbose=True):
    # Get tile IDS, image IDs and image targets
    tile_ids, image_ids, image_targets = get_tile_filename_info(data)
    
    #
    threshold_output = np.empty(
        len(thresholds),
        dtype=[
            ('threshold', float),
            ('f1', float),
            ('tpr', float),
            ('fpr', float)
        ]
    )
    
    for i in range(len(thresholds)):
        if verbose:
            print("Checking threshold", i)
        
        # Set up CNN learner
        learner = cnn_learner(data, model_type, metrics=accuracy).mixup()

        # Load trained model
        learner.load(trained_model)

        # Compute precition score
        tile_scores, tile_targets = learner.get_preds(ds_type=DatasetType.Valid)
        
        # Make tile predictions using prediction scores
        tile_preds = torch.argmax(tile_scores, 1)

        # Return predictions and true labels
        image_preds = image_output_to_array(predict_images(
            count_image_corrosion(image_ids, tile_preds, tile_targets, image_targets), thresholds[i])
        )
    
        image_metrics = error_metrics(image_preds['image_target'], image_preds['image_pred'])
        
        threshold_output['threshold'][i] = thresholds[i]
        threshold_output['f1'][i] = image_metrics['f1']
        threshold_output['tpr'][i] = image_metrics['tpr']
        threshold_output['fpr'][i] = image_metrics['fpr']
        
    
    optimal_threshold = thresholds[0]
    
    for i in range(1, len(thresholds)):
        if (
                (threshold_output['f1'][i] > threshold_output['f1'][i-1]) and
                ((tpr_lb is None) or (threshold_output['tpr'][i] > tpr_lb)) and
                ((fpr_ub is None) or (threshold_output['fpr'][i] < fpr_ub))
            ):
            optimal_threshold = thresholds[i]
    
    return optimal_threshold, threshold_output
