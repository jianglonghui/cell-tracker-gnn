import os

import yaml

from datamodules.preprocess_dataset import PreprocessDataset


def create_csv(input_images, input_seg, input_model, output_csv, ndim, min_cell_size=None):
    dict_path = input_model
    path_output = output_csv
    path_Seg_result = input_seg
    ds = PreprocessDataset(
        path=input_images,
        path_result=path_Seg_result,
        type_img="tif",
        type_masks="tif",
        ndim=ndim)
    if min_cell_size is not False:
        ds.correct_masks(min_cell_size)
    ds.preprocess_write_csv(path_to_write=path_output,
        dict_path=dict_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str, help='analysis configuration file')

    args = parser.parse_args()
    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    min_cell_size = conf['minimum cell size']
    input_images = conf['input image dir']
    input_segmentation = conf['segmentation dir']
    if not input_segmentation:
        input_segmentation = os.path.join(
            os.path.dirname(input_images),
            os.path.basename(input_images) + "_RES_SEG",
            )
    input_model = conf['all params dir']
    output_csv = conf['csv output dir']
    ndim = conf['dimension']

    create_csv(input_images, input_segmentation, input_model, output_csv, ndim, min_cell_size)
