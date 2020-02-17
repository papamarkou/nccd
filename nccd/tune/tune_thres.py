# %% Import packages

import numpy as np

import torch

from fastai.vision import DatasetType, cnn_learner, accuracy

from nccd.io import get_tile_filename_info, image_output_to_array
from nccd.summaries import count_image_corrosion, error_metrics
from nccd.predict import predict_images

# %% Function for computing tile prediction scores using trained model

def tune_thres(data, model_type, trained_model, thres, tpr_lb=None, fpr_ub=None, verbose=True):
    # Get tile IDS, image IDs and image targets
    tile_ids, image_ids, image_targets = get_tile_filename_info(data)

    # Allocate array that will hold error metrics for different thresholds
    thres_output = np.empty(
        len(thres),
        dtype=[
            ('thres', float),
            ('f1', float),
            ('tpr', float),
            ('fpr', float)
        ]
    )

    # Set up CNN learner
    learner = cnn_learner(data, model_type, metrics=accuracy).mixup()

    # Load trained model
    learner.load(trained_model)

    # Compute precition score
    tile_scores, tile_targets = learner.get_preds(ds_type=DatasetType.Valid)

    # Make tile predictions using prediction scores
    tile_preds = torch.argmax(tile_scores, 1)

    for i in range(len(thres)):
        if verbose:
            print(("Checking threshold {:" + str(max([len(str(t)) for t in thres])) + "} out of {}").format(i+1, len(thres)))

        # Return predictions and true labels
        image_preds = image_output_to_array(predict_images(
            count_image_corrosion(image_ids, tile_preds, tile_targets, image_targets), thres[i])
        )

        # Computer error metrics
        image_metrics = error_metrics(image_preds['image_target'], image_preds['image_pred'])

        # Store threshold and associated F1 score, TPR and FPR in array
        thres_output['thres'][i] = thres[i]
        thres_output['f1'][i] = image_metrics['f1']
        thres_output['tpr'][i] = image_metrics['tpr']
        thres_output['fpr'][i] = image_metrics['fpr']

    optimal_thres_idx = 0
    optimal_thres = thres[0]

    # Find optimal threshold
    for i in range(1, len(thres)):
        if (
                (thres_output['f1'][i] > thres_output['f1'][i-1]) and
                ((tpr_lb is None) or (thres_output['tpr'][i] > tpr_lb)) and
                ((fpr_ub is None) or (thres_output['fpr'][i] < fpr_ub))
            ):
            optimal_thres_idx = i
            optimal_thres = thres[i]

    return optimal_thres, optimal_thres_idx, thres_output
