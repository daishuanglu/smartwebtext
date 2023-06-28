import h5py
import os
import numpy as np

main_dir = "D:/video_datasets/A2D/a2d_annotation_with_instances/a2d_annotation_with_instances"
fpath = os.path.join(main_dir, '_0djE279Srg/00030.h5')

with h5py.File(fpath, 'r') as mat_file:
    for k in mat_file.keys():
        print(k)
        print(mat_file[k])
