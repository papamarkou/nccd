# %% Function for making image predictions using tile predictions

def predict_images(image_dict, threshold):
    output_dict = dict()

    for k, v in image_dict.items():
        output_dict[k] = [v[0], 1 if (v[0] > threshold) else 0, v[1]]

    return output_dict

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
