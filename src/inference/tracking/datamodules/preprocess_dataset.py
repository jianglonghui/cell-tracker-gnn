import glob
import os
import os.path as op
import warnings
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import io
from skimage.measure import regionprops

from .base_preprocess_dataset import BasePreprocessDataset

warnings.filterwarnings("ignore")

from skimage.morphology import label


class PreprocessDataset(BasePreprocessDataset):
    """Example dataset class for loading images from folder."""

    def __init__(self,
                 path: str,
                 path_result: str,
                 type_img: str,
                 type_masks: str,
                 ndim: int,
                 ):
        assert ndim in [2, 3], f'dimension has to be 2(D) or 3(D).'
        self.ndim = ndim

        self.__path = path
        self.__path_result = path_result
        self.__images = []
        self.__results = []
        self.__flag_new_roi = False
        self.__global_delta = {}

        dir_img = path
        dir_results = path_result

        assert os.path.exists(dir_img), f"Input image dir ({dir_img}) does not exist"
        self.__images = sorted(glob.glob(os.path.join(dir_img, "*." + type_img)), key=lambda x: int(os.path.splitext(x)[0][-3:]))
        assert self.__images, f"Couldn't find images in {dir_img} of type {type_img}"

        assert os.path.exists(dir_results), f"Input segmenteation dir ({dir_results}) does not exist"
        self.__results = sorted(glob.glob(os.path.join(dir_results, "*." + type_masks)), key=lambda x: int(os.path.splitext(x)[0][-3:]))
        assert self.__results, f"Couldn't find images in {dir_results} of type {type_masks}"

    def __getitem__(self, idx):

        im_path, image = None, None
        im_path = self.__images[idx]
        image = np.stack(io.imread(im_path))
        assert im_path, f'Input image index {idx} does not exist.'
        assert image is not None, f'Read input image of index {idx} is None.'

        result_path, result = None, None
        result_path = self.__results[idx]
        result = np.stack(io.imread(result_path))
        assert result_path, f'Segmentation image index {idx} does not exist.'
        assert result is not None, f'Read segmentation image of index {idx} is None.'

        assert image.shape == result.shape, f"Input image size is not equal to segmentation image size"

        im_num, result_num = None, None

        im_num = os.path.splitext(im_path)[0][-3:]
        result_num = os.path.splitext(result_path)[0][-3:]

        assert im_num == result_num, f"Image number ({im_num}) is not equal to result number ({result_num})"

        return image, result, im_path, result_path

    def __len__(self):
        return len(self.__images)

    def _padding(self, img):
        desired_size = {}
        if self.__flag_new_roi:
            for key, value in self.__global_delta.items():
                desired_size[key] = value
        if self.ndim == 3:
            delta_depth = desired_size['depth'] - img.shape[0]
            delta_row = desired_size['row'] - img.shape[1]
            delta_col = desired_size['col'] - img.shape[2]
            pad_depth = delta_depth // 2
        else:
            delta_row = desired_size['row'] - img.shape[0]
            delta_col = desired_size['col'] - img.shape[1]
        pad_top = delta_row // 2
        pad_left = delta_col // 2

        if self.ndim == 3:
            image = np.pad(img,
                           ((pad_depth, delta_depth - pad_depth),
                            (pad_top, delta_row - pad_top),
                            (pad_left, delta_col - pad_left)),
                           'constant', constant_values=np.ones((3, 2)) * self.__pad_value)
            if self.__flag_new_roi:
                image = torch.nn.functional.interpolate(torch.from_numpy(image[None, None, ...]),
                                                        size=(self.__roi_model['depth'],
                                                        self.__roi_model['row'],
                                                        self.__roi_model['col']),
                                                        mode='trilinear')
                image = image.numpy().squeeze()
        else:
            image = cv2.copyMakeBorder(img, pad_top, delta_row - pad_top, pad_left, delta_col - pad_left,
                                       cv2.BORDER_CONSTANT, value=self.__pad_value)
            if self.__flag_new_roi:
                image = cv2.resize(image, dsize=(self.__roi_model['col'], self.__roi_model['row']))
        return image

    def _extract_freature_metric_learning(self, bbox, img, seg_mask, ind, normalize_type='MinMaxCell'):
        if self.ndim == 3:
            min_depth_bb, min_row_bb, min_col_bb, \
            max_depth_bb, max_row_bb, max_col_bb = bbox
            img_patch = img[min_depth_bb:max_depth_bb,
                            min_row_bb:max_row_bb,
                            min_col_bb:max_col_bb]
            msk_patch = seg_mask[min_depth_bb:max_depth_bb,
                                min_row_bb:max_row_bb,
                                min_col_bb:max_col_bb] != ind
        else:
            min_row_bb, min_col_bb, max_row_bb, max_col_bb = bbox
            img_patch = img[min_row_bb:max_row_bb, min_col_bb:max_col_bb]
            msk_patch = seg_mask[min_row_bb:max_row_bb, min_col_bb:max_col_bb] != ind
        img_patch[msk_patch] = self.__pad_value
        img_patch = img_patch.astype(np.float32)

        if normalize_type == 'regular':
            img = self._padding(img_patch) / self.max_img
        elif normalize_type == 'MinMaxCell':
            not_msk_patch = np.logical_not(msk_patch)
            img_patch[not_msk_patch] = (img_patch[not_msk_patch] - self.__min_cell) / (self.__max_cell - self.__min_cell)
            img = self._padding(img_patch)
        else:
            assert False, "Not supported this type of normalization"

        img = torch.from_numpy(img).float()
        with torch.no_grad():
            embedded_img = self.__embedder(self.__trunk(img[None, None, ...]))

        return embedded_img.numpy().squeeze()

    def correct_masks(self, min_cell_size):
        n_changes = 0
        for ind_data in range(self.__len__()):
            per_cell_change = False
            per_mask_change = False

            img, result, im_path, result_path = self[ind_data]
            res_save = result.copy()
            labels_mask = result.copy()
            while True:
                bin_mask = labels_mask > 0
                re_label_mask = label(bin_mask)
                un_labels, counts = np.unique(re_label_mask, return_counts=True)

                if np.any(counts < min_cell_size):
                    per_mask_change = True

                    first_label_ind = np.argwhere(counts < min_cell_size)
                    if first_label_ind.size > 1:
                        first_label_ind = first_label_ind.squeeze()[0]
                    first_label_num = un_labels[first_label_ind]
                    labels_mask[re_label_mask == first_label_num] = 0
                else:
                    break
            bin_mask = (labels_mask > 0) * 1.0
            result = np.multiply(result, bin_mask)
            if not np.all(np.unique(result) == np.unique(res_save)):
                warnings.warn(
                    f"pay attention! the labels have changed from {np.unique(res_save)} to {np.unique(result)}")


            for ind, id_res in enumerate(np.unique(result)):
                if id_res == 0:
                    continue
                bin_mask = (result == id_res).copy()
                while True:
                    re_label_mask = label(bin_mask)
                    un_labels, counts = np.unique(re_label_mask, return_counts=True)

                    if np.any(counts < min_cell_size):
                        per_cell_change = True

                        first_label_ind = np.argwhere(counts < min_cell_size)
                        if first_label_ind.size > 1:
                            first_label_ind = first_label_ind.squeeze()[0]
                        first_label_num = un_labels[first_label_ind]
                        curr_mask = np.logical_and(result == id_res, re_label_mask == first_label_num)
                        bin_mask[curr_mask] = False
                        result[curr_mask] = 0.0
                    else:
                        break
                while True:
                    re_label_mask = label(bin_mask)
                    un_labels, counts = np.unique(re_label_mask, return_counts=True)
                    if un_labels.shape[0] > 2:
                        per_cell_change = True
                        n_changes += 1
                        first_label_ind = np.argmin(counts)
                        if first_label_ind.size > 1:
                            first_label_ind = first_label_ind.squeeze()[0]
                        first_label_num = un_labels[first_label_ind]
                        curr_mask = np.logical_and(result == id_res, re_label_mask == first_label_num)
                        bin_mask[curr_mask] = False
                        result[curr_mask] = 0.0
                    else:
                        break
            if not np.all(np.unique(result) == np.unique(res_save)):
                warnings.warn(
                    f"pay attention! the labels have changed from {np.unique(res_save)} to {np.unique(result)}")
            if per_cell_change or per_mask_change:
                n_changes += 1
                res1 = (res_save > 0) * 1.0
                res2 = (result > 0) * 1.0
                n_pixels = np.abs(res1 - res2).sum()
                print(f"per_mask_change={per_mask_change}, per_cell_change={per_cell_change}, number of changed pixels: {n_pixels}")
                io.imsave(result_path, result.astype(np.uint16), compress=6)

        print(f"number of detected changes: {n_changes}")


    def _find_min_max_and_roi(self):
        global_min = 2 ** 16 - 1
        global_max = 0
        global_delta = defaultdict(int)
        delta = {}
        min_bb = {}
        max_bb = {}
        counter = 0
        for ind_data in range(self.__len__()):
            img, result, im_path, result_path = self[ind_data]
            if img is None or result is None:
                print('*' * 20 + 'We have None' + 20 * '*')
            for ind, id_res in enumerate(np.unique(result)):
                if id_res == 0:
                    continue

                properties = regionprops(np.uint8(result == id_res), img)[0]
                if self.ndim == 3:
                    min_bb['depth'], min_bb['row'], min_bb['col'], \
                    max_bb['depth'], max_bb['row'], max_bb['col'] = properties.bbox
                else:
                    min_bb['row'], min_bb['col'], max_bb['row'], max_bb['col'] = properties.bbox

                for key in max_bb:
                    delta[key] = np.abs(max_bb[key] - min_bb[key])

                chk = False
                for key, value in delta.items():
                    chk |= value > self.__roi_model[key]
                counter += chk
                print(f"bigger ROI: {delta}")

                for key, value in delta.items():
                    global_delta[key] = max(global_delta[key], value)

            res_bin = result != 0
            min_curr = img[res_bin].min()
            max_curr = img[res_bin].max()

            global_min = min(global_min, min_curr)
            global_max = max(global_max, max_curr)

        print(counter)
        for key, value in global_delta.items():
            print(f"global_delta_{key}: {value}")
            self.__global_delta[key] = value


        self.__min_cell = global_min
        self.__max_cell = global_max


    def preprocess_features_w_metric_learning(self, dict_path):
        if self.ndim == 3:
            from src_metric_learning.modules.resnet_3d.resnet import (
                MLP, set_model_architecture)
        else:
            from src_metric_learning.modules.resnet_2d.resnet import (
                MLP, set_model_architecture)
        dict_params = torch.load(dict_path)

        self.__roi_model = dict_params['roi']
        self._find_min_max_and_roi()
        self.__flag_new_roi = False
        for key, value in self.__global_delta.items():
            self.__flag_new_roi |= value > self.__roi_model[key]

        if self.__flag_new_roi:
            for key, value in self.__global_delta.items():
                self.__global_delta[key] = max(value, self.__roi_model[key])
            print("Assign new region of interest")
            print(f"old ROI: {self.__roi_model}, new: {self.__global_delta}")
        else:
            print("We don't assign new region of interest - use the old one")

        self.__pad_value = dict_params['pad_value']
        # models params
        model_name = dict_params['model_name']
        mlp_dims = dict_params['mlp_dims']
        mlp_normalized_features = dict_params['mlp_normalized_features']
        # models state_dict
        trunk_state_dict = dict_params['trunk_state_dict']
        embedder_state_dict = dict_params['embedder_state_dict']

        trunk = set_model_architecture(model_name)
        trunk.load_state_dict(trunk_state_dict)
        self.__trunk = trunk
        self.__trunk.eval()

        embedder = MLP(mlp_dims, normalized_feat=mlp_normalized_features)
        embedder.load_state_dict(embedder_state_dict)
        self.__embedder = embedder
        self.__embedder.eval()

        if self.ndim == 3:
            cols = ["seg_label",
                    "frame_num",
                    "area",
                    "min_depth_bb", "min_row_bb", "min_col_bb",
                    "max_depth_bb", "max_row_bb", "max_col_bb",
                    "centroid_depth", "centroid_row", "centroid_col",
                    "major_axis_length", "minor_axis_length",
                    "max_intensity", "mean_intensity", "min_intensity",
                    ]
        else:
            cols = ["seg_label",
                    "frame_num",
                    "area",
                    "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
                    "centroid_row", "centroid_col",
                    "major_axis_length", "minor_axis_length",
                    "max_intensity", "mean_intensity", "min_intensity",
                    ]

        cols_resnet = [f'feat_{i}' for i in range(mlp_dims[-1])]
        cols += cols_resnet

        for ind_data in range(self.__len__()):
            img, result, im_path, result_path = self[ind_data]
            if img is None or result is None:
                print('*' * 20 + 'We have None' + 20 * '*')
            im_num = im_path.split(".")[-2][-3:]
            result_num = result_path.split(".")[-2][-3:]
            assert im_num == result_num, f"Image number ({im_num}) is not equal to result number ({result_num})"

            num_labels = np.unique(result).shape[0] - 1

            df = pd.DataFrame(index=range(num_labels), columns=cols)

            for ind, id_res in enumerate(np.unique(result)):
                # Color 0 is assumed to be background or artifacts
                row_ind = ind - 1
                if id_res == 0:
                    continue

                # extracting statistics using regionprops
                properties = regionprops(np.uint8(result == id_res), img)[0]

                embedded_feat = self._extract_freature_metric_learning(properties.bbox, img.copy(), result.copy(), id_res)
                df.loc[row_ind, cols_resnet] = embedded_feat
                df.loc[row_ind, "seg_label"] = id_res

                df.loc[row_ind, "area"] = properties.area

                if self.ndim == 3:
                    df.loc[row_ind, "min_depth_bb"], df.loc[row_ind, "min_row_bb"], \
                    df.loc[row_ind, "min_col_bb"], df.loc[row_ind, "max_depth_bb"], \
                    df.loc[row_ind, "max_row_bb"], df.loc[row_ind, "max_col_bb"] = properties.bbox
                    df.loc[row_ind, "centroid_depth"], df.loc[row_ind, "centroid_row"], df.loc[row_ind, "centroid_col"] = \
                        properties.centroid[0].round().astype(np.int16), \
                        properties.centroid[1].round().astype(np.int16), \
                        properties.centroid[2].round().astype(np.int16)
                else:
                    df.loc[row_ind, "min_row_bb"], df.loc[row_ind, "min_col_bb"], \
                    df.loc[row_ind, "max_row_bb"], df.loc[row_ind, "max_col_bb"] = properties.bbox
                    df.loc[row_ind, "centroid_row"], df.loc[row_ind, "centroid_col"] = \
                        properties.centroid[0].round().astype(np.int16), \
                        properties.centroid[1].round().astype(np.int16)

                df.loc[row_ind, "major_axis_length"], df.loc[row_ind, "minor_axis_length"] = \
                    properties.major_axis_length, properties.minor_axis_length

                df.loc[row_ind, "max_intensity"], df.loc[row_ind, "mean_intensity"], df.loc[row_ind, "min_intensity"] = \
                    properties.max_intensity, properties.mean_intensity, properties.min_intensity


            df.loc[:, "frame_num"] = int(im_num)

            if df.isnull().values.any():
                warnings.warn("Pay Attention! there are Nan values!")

            yield df, im_num

    def preprocess_write_csv(self, path_to_write, dict_path):
        full_dir = os.path.join(path_to_write, "csv")
        os.makedirs(full_dir, exist_ok=True)
        for df, im_num in self.preprocess_features_w_metric_learning(dict_path):
            file_path = op.join(full_dir, f"frame_{im_num}.csv")
            print(f"save file to : {file_path}")
            df.to_csv(file_path, index=False)
        print(f"files were saved to : {full_dir}")
