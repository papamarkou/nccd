# %% Parse single line from tile filename to get tile  IDs, image IDs and image targets
    
def parse_default_tile_filename(filename, corroded_label='c'):
    image_info = filename.rsplit("_")[0]
    image_id =image_info[:-1]
    image_target = 1 if (image_info[-1] == corroded_label) else 0
    return image_id, image_target

# %% Function for getting tile IDS, image IDs and image targets

def get_tile_filename_info(data, parse_filename=parse_default_tile_filename):
    tile_ids = []
    image_ids = []
    image_targets = []

    for item in data.valid_ds.items:
        tile_ids.append(item.stem)
        image_id, image_target = parse_filename(item.stem)
        image_ids.append(image_id)
        image_targets.append(image_target)

    return tile_ids, image_ids, image_targets
