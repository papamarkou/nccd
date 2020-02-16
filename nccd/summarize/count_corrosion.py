# %% Function for counting number of corroded tiles per image

def count_image_corrosion(image_ids, tile_preds, tile_targets, image_targets, threshold):
    image_dict = dict()

    for image_id, image_target in zip(image_ids, image_targets):
        if image_id not in image_dict:
            image_dict[image_id] = [0, image_target]

    for image_id, tile_pred, tile_target in zip(image_ids, tile_preds, tile_targets):
        if tile_pred.item() == 1:
            image_dict[image_id][0] = image_dict[image_id][0] + 1

    return image_dict
