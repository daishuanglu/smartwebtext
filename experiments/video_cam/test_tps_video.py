import pims
import numpy as np



from utils import image_utils
from utils.train_utils import device
from experiments.tps_energy import tps_transform_energy


def get_grid_points(mask, grid_size):
    mask_height, mask_width = mask.shape

    # Generate grid coordinates using np.meshgrid
    x_grid, y_grid = np.meshgrid(np.arange(grid_size // 2, mask_width, grid_size),
                                 np.arange(grid_size // 2, mask_height, grid_size))
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    # Calculate indices of grid points within the mask region
    mask_indices = np.nonzero(mask[y_grid, x_grid])

    # Obtain grid point coordinates
    grid_points = list(zip(x_grid[mask_indices], y_grid[mask_indices]))

    return grid_points


def pw_tps_energy(s_color_mask, t_color_mask, alpha=0.0, grid_size=16):
    s_region_ids = image_utils.color_map_to_region_id_map(s_color_mask)
    t_region_ids = image_utils.color_map_to_region_id_map(t_color_mask)
    s_uniq_ids = np.unique(s_region_ids)
    t_uniq_ids = np.unique(t_region_ids)
    pw_tps_e = np.zeros(len(s_uniq_ids), len(t_uniq_ids))
    for t_id in t_uniq_ids:
        t_mask = t_region_ids == t_id
        t_grid = get_grid_points(t_mask, grid_size)
        for s_id in s_uniq_ids:
            s_mask = s_region_ids == s_id
            s_grid = get_grid_points(s_mask, grid_size)
            pw_tps_e[s_id, t_id] = tps_transform_energy(s_grid, t_grid, alpha, device)
    return pw_tps_e


def match_regions(v):
    for i in range(len(v)-1):
        pw_tps_e = pw_tps_energy(v[i], v[i+1], alpha=0.0, grid_size=16)

    return


weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

person_float = transforms(person_int)

model = keypointrcnn_resnet50_fpn(weights=weights, progress=False)
model = model.eval()

outputs = model([person_float])
print(outputs)



SAM_VIDEO_PATH = 'D:/video_datasets/A2D/A2D_main_1_0/Release/SAMclips320H/q01o1J7zy5A.mp4'
vf = pims.Video(SAM_VIDEO_PATH)
#for frame in vf:
#    region_id = image_utils.color_map_to_region_id_map(frame)
#    for i in np.unique(region_id):
#        mask = region_id == i
#        generate_grid(mask)
