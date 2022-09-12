import argparse
import os
import pdb
import re
import tempfile
from glob import glob
from itertools import product

import numpy as np
import tensorflow as tf
import yaml
from keras.models import load_model
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, median_filter
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries, slic, watershed
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tifffile import imread, imsave
from track import track_main

np.random.seed(42)

def normalize(x, pmin=3, pmax=99.8, axis = None, clip=False):
    mi = np.percentile(x, pmin, axis = axis, keepdims=True).astype(np.float32)
    ma = np.percentile(x, pmax, axis = axis, keepdims=True).astype(np.float32)
    x = x.astype(np.float32)
    eps = 1.e-20
    y = (1. * x - mi) / (ma - mi+eps)
    if clip:
        y = np.clip(y, 0, 1)
    return y

def tile_iterator(im,
                 blocksize = (64, 64),
                 padsize = (64,64),
                 mode = "constant",
                 verbose = False):
    """

    iterates over padded tiles of an ND image
    while keeping track of the slice positions

    Example:
    --------
    im = np.zeros((200,200))
    res = np.empty_like(im)

    for padded_tile, s_src, s_dest in tile_iterator(im,
                              blocksize=(128, 128),
                              padsize = (64,64),
                              mode = "wrap"):

        #do something with the tile, e.g. a convolution
        res_padded = np.mean(padded_tile)*np.ones_like(padded_tile)

        # reassemble the result at the correct position
        res[s_src] = res_padded[s_dest]



    Parameters
    ----------
    im: ndarray
        the input data (arbitrary dimension)
    blocksize:
        the dimension of the blocks to split into
        e.g. (nz, ny, nx) for a 3d image
    padsize:
        the size of left and right pad for each dimension
    mode:
        padding mode, like numpy.pad
        e.g. "wrap", "constant"...

    Returns
    -------
        tile, slice_src, slice_dest

        tile[slice_dest] is the tile in im[slice_src]

    """

    if not(im.ndim == len(blocksize) ==len(padsize)):
        raise ValueError("im.ndim (%s) != len(blocksize) (%s) != len(padsize) (%s)"
                         %(im.ndim , len(blocksize) , len(padsize)))

    subgrids = tuple([int(np.ceil(1.*n/b)) for n,b in zip(im.shape, blocksize)])


    #if the image dimension are not divible by the blocksize, pad it accordingly
    pad_mismatch = tuple([(s*b-n) for n,s, b in zip(im.shape,subgrids,blocksize)])

    if verbose:
        print("tile padding... ")

    im_pad = np.pad(im,[(p,p+pm) for pm,p in zip(pad_mismatch,padsize)], mode = mode)

    # iterates over cartesian product of subgrids
    for i,index in enumerate(product(*[range(sg) for sg in subgrids])):
        # the slices
        # if verbose:
        #     print("tile %s/%s"%(i+1,np.prod(subgrids)))

        # dest[s_output] is where we will write to
        s_input = tuple([slice(i*b,(i+1)*b) for i,b in zip(index, blocksize)])



        s_output = tuple([slice(p,-p-pm*(i==s-1)) for pm,p,i,s in zip(pad_mismatch,padsize, index, subgrids)])


        s_output = tuple([slice(p,b+p-pm*(i==s-1)) for b,pm,p,i,s in zip(blocksize,pad_mismatch,padsize, index, subgrids)])


        s_padinput = tuple([slice(i*b,(i+1)*b+2*p) for i,b,p in zip(index, blocksize, padsize)])
        padded_block = im_pad[s_padinput]



        yield padded_block, s_input, s_output


def apply(model, x, tile_size, div = 4):
    def _make_divisible(n,div):
        return int(np.ceil(n/div))*div

    assert x.ndim ==3
    tile_size = tuple(s if t==-1 else t for s,t in zip(x.shape,tile_size))
    tile_size = tuple(_make_divisible(t, div) for t in tile_size)

    pad_size = tuple(0 if s<=t else 64 for s,t in zip(x.shape,tile_size))

    x = normalize(x,2,99.7)

    res = np.empty_like(x)

    for tile, s_src, s_dest in tile_iterator(x,
                                             blocksize=tile_size,
                                             padsize = pad_size,
                                             mode = "reflect"):
        X = tile[np.newaxis,...,np.newaxis]
        if len(model.inputs)>1:
            X = [X]*len(model.inputs)

        y = model.predict(X)[0,...,0]
        res[s_src] = y[s_dest]

    return res

def dice_coef(y_true,y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true*y_pred))
    return (2*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def dice_loss(y_true,y_pred):
    return 1-dice_coef(y_true,y_pred)


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto( intra_op_parallelism_threads=22,
                        inter_op_parallelism_threads=22,
                        allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    parser = argparse.ArgumentParser()

    parser.add_argument('conf', type=str, help='analysis configuration file')

    args = parser.parse_args()

    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    with open(conf['seg conf'], 'r') as f:
        seg_conf = yaml.safe_load(f)

    default_conf = {
        'threshold_abs': 0.3,
        'min_distance': 4,
        'tile_size': [-1,512,256],
        }

    model_center = load_model(seg_conf['model center'])
    model_mask = load_model(seg_conf['model prob'])

    files = sorted(glob(os.path.join(conf['input image dir'], "*.tif")))
    outdirseg = conf['segmentation dir']
    #outdirtrack = conf['tracking dir']
    for key, value in default_conf.items():
        if key not in seg_conf:
            seg_conf[key] = value
    os.makedirs(outdirseg, exist_ok = True)
    fl = len(files)

    for i,f in enumerate(files):
        tp = re.findall("t([0-9]+?)\.",os.path.basename(f))
        if len(tp)==0:
            tp = os.path.basename(f)
        else:
            tp = tp[0]

        save_path = os.path.join(outdirseg, "mask%s.tif"%tp)

        if seg_conf['write over'] is False and os.path.exists(save_path):
            print("prediction %s/%s is already done."%(i+1, fl) )
            continue
        print("predicting %s/%s"%(i+1, fl) )

        x = imread(f)
        u_center = apply(model_center, x, tile_size=seg_conf['tile_size'])
        u_mask= apply(model_mask, np.pad(x,((2,2),(0,0),(0,0)), mode="reflect"), tile_size=seg_conf['tile_size'])[2:-2]
        peak_indices = peak_local_max(u_center, min_distance= seg_conf['min_distance'], threshold_abs=seg_conf['threshold_abs'], exclude_border=1)
        peaks = np.zeros_like(u_center, dtype=bool)
        peaks[tuple(peak_indices.T)] = True

        img = median_filter(x,[1,3,5])

        markers = ndi.label(peaks)[0]
        mask = u_mask>.3
        ref_label = watershed(-u_mask, markers, mask=mask)

        segments = slic(img, n_segments=1400, compactness=0.1, multichannel = False, sigma = 0)
        ref_label_x, ref_label_y, ref_label_z = segments.shape

        labels = (np.array(segments, dtype=np.int32)).reshape(ref_label.shape)

        final_labels = np.empty([ref_label_x,ref_label_y,ref_label_z])
        dictionary_labels = {}
        for l in range (ref_label_x):
            for m in range(ref_label_y):
                for n in range(ref_label_z):
                    if not dictionary_labels.get(labels[l][m][n]):
                        dictionary_labels[labels[l][m][n]] = [ref_label[l][m][n]]
                    else:
                        dictionary_labels[labels[l][m][n]].append(ref_label[l][m][n])
        for l in dictionary_labels.keys():
            L = dictionary_labels[l]
            d = {}

            for i in L:
                if not d.get(i):
                    d[i] = 1
                else:
                    d[i] += 1
            ma_index = 0
            ma_value = 0
            for i in d.keys():
                if (d[i] > ma_value):
                    ma_index = i
                    ma_value = d[i]

            dictionary_labels[l] = ma_index

        for l in range (ref_label_x):
            for m in range(ref_label_y):
                for n in range(ref_label_z):
                    final_labels[l][m][n] = dictionary_labels[labels[l][m][n]]
                    if (ref_label[l][m][n] == 0):
                        ref_label[l][m][n] = final_labels[l][m][n]

        imsave(save_path, ref_label.astype(np.uint16))
    with tempfile.TemporaryDirectory() as tmp_dir:
        outdirtrack = tmp_dir
        track_main(outdirseg, outdirtrack)
