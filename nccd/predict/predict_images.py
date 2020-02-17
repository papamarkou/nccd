# %% Function for making image predictions using tile predictions

def predict_images(image_dict, threshold):
    output_dict = dict()

    for k, v in image_dict.items():
        output_dict[k] = [v[0], 1 if (v[0] >= threshold) else 0, v[1]]

    return output_dict
