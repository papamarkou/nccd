# %% Import packages

import numpy as np

# %% Function for placing tile output to a numpy array
    
def tile_output_to_array(tile_ids, image_ids, tile_scores, tile_preds, tile_targets, image_targets):
    tile_output = np.empty(
        len(tile_ids),
        dtype=[
            ('tile_id', '<U{0}'.format(np.max([len(id) for id in tile_ids]))),
            ('image_id', '<U{0}'.format(np.max([len(id) for id in image_ids]))),
            ('tile_score', float),
            ('tile_pred', int),
            ('tile_target', int),
            ('image_target', int)
        ]
    )
    
    tile_output['tile_id'] = tile_ids
    tile_output['image_id'] = image_ids
    tile_output['tile_score'] = tile_scores[: , 1].clone().detach().numpy()
    tile_output['tile_pred'] = tile_preds.clone().detach().cpu()
    tile_output['tile_target'] = tile_targets.clone().detach().cpu()
    tile_output['image_target'] = image_targets
    
    return tile_output

# %% Function for placing image output to a numpy array

def image_corrosion_output_to_array(image_dict):
    image_corrosion_output = np.empty(
        len(image_dict),
        dtype=[
            ('image_id', '<U{0}'.format(np.max([len(k) for k in image_dict.keys()]))),
            ('image_corrosion_count', int),
            ('image_target', int)
        ]
    )

    for i, (k, v) in enumerate(image_dict.items()):
        image_corrosion_output[i] = k, v[0], v[1]
        
    return image_corrosion_output

# %% Function for placing image output to a numpy array

def image_output_to_array(image_dict):
    image_output = np.empty(
        len(image_dict),
        dtype=[
            ('image_id', '<U{0}'.format(np.max([len(k) for k in image_dict.keys()]))),
            ('image_corrosion_count', int),
            ('image_pred', int),
            ('image_target', int)
        ]
    )

    for i, (k, v) in enumerate(image_dict.items()):
        image_output[i] = k, v[0], v[1], v[2]
        
    return image_output

# %% Function for placing image metrics to a numpy array
    
def image_metrics_to_array(image_dict):
    image_metrics = np.empty(
        len(image_dict),
        dtype=[('metric_name', '<U{0}'.format(np.max([len(k) for k in image_dict.keys()]))), ('metric_value', float)]
    )

    for i, (k, v) in enumerate(image_dict.items()):
        image_metrics[i] = k, v
        
    return image_metrics
