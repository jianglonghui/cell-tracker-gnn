import warnings

from torch.utils.data import Dataset

warnings.filterwarnings("always")


class BasePreprocessDataset(Dataset):
    """Example dataset class for loading images from folder."""

    def __init__(self,
                 path: str,
                 path_result: str,
                 type_img: str,
                 type_masks: str,
                 ndim: int,
                 ):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _padding(self, img):
        """
        Args:
            img (ndarray): a single input image.

        Returns:
            ndarray: ndarray of the processed image.
        """
        raise NotImplementedError

    def _extract_freature_metric_learning(self, bbox, img, seg_mask, ind, normalize_type='MinMaxCell'):
        """
        Args:
            bbox (tuple): bbox from regionprops
            img (ndarray): a single input image.
            seg_mask (ndarray): the segmentation result for a single input image.
            ind (int): the index of the input image.
            normalize_type (:obj:`string`, optional): type of normalizing()

        Returns:
            ndarray: embedded image
        """
        raise NotImplementedError

    def correct_masks(self, min_cell_size):
        """
        Args:
            min_cell_size (float): minimum detection limit

        Returns:

        Note:
            this overwrites original images.
        """
        raise NotImplementedError

    def _find_min_max_and_roi(self):
        raise NotImplementedError

    def preprocess_features_w_metric_learning(self, dict_path):
        """
        Args:
            dict_path: path to params of trained torch model.

        Yields:
            DataFrame: preprocessed features
            int: index of image
        """
        raise NotImplementedError

    def preprocess_write_csv(self, path_to_write, dict_path):
        raise NotImplementedError
