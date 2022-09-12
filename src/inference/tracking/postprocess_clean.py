import os
import os.path as osp
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from skimage import io

warnings.filterwarnings("ignore")
from datamodules.postprocess_dataset import Postprocess

if __name__== "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str, help='analysis configuration file')

    args = parser.parse_args()
    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    modality = conf['dimension']
    assert modality == 2 or modality == 3
    is_3d = modality == 3
    directed = True
    merge_operation = 'AND'
    path_Seg_result = conf['segmentation dir']
    path_inference_output = conf['inference dir']
    input_images = conf['input image dir']
    ndim = conf['dimension']
    path_tra_result = conf['tracking dir']

    if path_inference_output:
        input_images = conf['input image dir']
        input_segmentation = os.path.join(
            os.path.dirname(input_images),
            os.path.basename(input_images) + "_RES_Inference",
            )

    if path_inference_output:
        input_images = conf['input image dir']
        input_segmentation = os.path.join(
            os.path.dirname(input_images),
            os.path.basename(input_images) + "_RES_SEG",
            )


    with tempfile.TemporaryDirectory() as tmp_dir:
        path_inference_output = path_inference_output
        pp = Postprocess(ndim=ndim,
                        type_masks='tif', merge_operation=merge_operation,
                        decision_threshold=0.5,
                        path_inference_output=path_inference_output, center_coord=False,
                        directed=directed,
                        path_seg_result=path_Seg_result,
                        path_tra_result=path_tra_result)
        all_frames_traject, trajectory_same_label, df_trajectory, str_track = pp.create_trajectory()
        pp.fill_mask_labels(debug=False)
