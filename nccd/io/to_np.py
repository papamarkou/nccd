# %% Import packages

import numpy as np

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

# %% Function for placing image output to a numpy array

def image_corrosion_output_to_array(image_dict):
    image_corrosion_output = np.empty(
        len(image_dict),
        dtype=[('col1', '<U{0}'.format(np.max([len(k) for k in image_dict.keys()]))), ('col2', int), ('col3', int)]
    )

    for i, (k, v) in enumerate(image_dict.items()):
        image_corrosion_output[i] = k, v[0], v[1]
        
    return image_corrosion_output

# %% Function for placing image output to a numpy array

def image_output_to_array(image_dict):
    image_output = np.empty(
        len(image_dict),
        dtype=[
            ('col1', '<U{0}'.format(np.max([len(k) for k in image_dict.keys()]))),
            ('col2', int),
            ('col3', int),
            ('col4', int)
        ]
    )

    for i, (k, v) in enumerate(image_dict.items()):
        image_output[i] = k, v[0], v[1], v[2]
        
    return image_output
