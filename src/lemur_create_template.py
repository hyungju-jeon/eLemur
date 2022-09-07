
import os
import sys
sys.path.append('hyungju/code/fenglab')
import json
import glob
import copy
import pathlib
import scipy.io
import scipy.ndimage
import natsort
import itertools
import lap
import cv2
import pandas as pd
import re
import math
import numpy as np
import ants
import shutil
import fnmatch

from zimg import *
from utils import io
from utils import img_util
from utils import nim_roi
from utils import region_annotation
from utils.logger import setup_logger
from tempfile import mktemp



def build_template(
    initial_template=None,
    image_list=None,
    iterations=3,
    gradient_step=0.2,
    blending_weight=0.75,
    weights=None,
    **kwargs
):
    """
    Estimate an optimal template from an input image_list
    ANTsR function: N/A
    Arguments
    ---------
    initial_template : ANTsImage
        initialization for the template building
    image_list : ANTsImages
        images from which to estimate template
    iterations : integer
        number of template building iterations
    gradient_step : scalar
        for shape update gradient
    blending_weight : scalar
        weight for image blending
    weights : vector
        weight for each input image
    kwargs : keyword args
        extra arguments passed to ants registration
    Returns
    -------
    ANTsImage
    Example
    -------
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data('r16') )
    >>> image2 = ants.image_read( ants.get_ants_data('r27') )
    >>> image3 = ants.image_read( ants.get_ants_data('r85') )
    >>> timage = ants.build_template( image_list = ( image, image2, image3 ) ).resample_image( (45,45))
    >>> timagew = ants.build_template( image_list = ( image, image2, image3 ), weights = (5,1,1) )
    """
    if "type_of_transform" not in kwargs:
        type_of_transform = "SyN"
    else:
        type_of_transform = kwargs.pop("type_of_transform")

    if weights is None:
        weights = np.repeat(1.0 / len(image_list), len(image_list))
    weights = [x / sum(weights) for x in weights]
    if initial_template is None:
        initial_template = image_list[0] * 0
        for i in range(len(image_list)):
            temp = image_list[i] * weights[i]
            temp = resample_image_to_target(temp, initial_template)
            initial_template = initial_template + temp

    xavg = initial_template.clone()
    for i in range(iterations):
        for k in range(len(image_list)):
            w1 = registration(
                xavg, image_list[k], type_of_transform=type_of_transform, **kwargs
            )
            if k == 0:
                wavg = iio.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = w1["warpedmovout"] * weights[k]
            else:
                wavg = wavg + iio.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = xavgNew + w1["warpedmovout"] * weights[k]
        print(wavg.abs().mean())
        wscl = (-1.0) * gradient_step
        wavg = wavg * wscl
        wavgfn = mktemp(suffix=".nii.gz")
        iio.image_write(wavg, wavgfn)
        xavg = apply_transforms(xavgNew, xavgNew, wavgfn)
        if blending_weight is not None:
            xavg = xavg * blending_weight + utils.iMath(xavg, "Sharpen") * (
                1.0 - blending_weight
            )

    return xavg

if __name__ == "__main__":

    folder = '/home/hyungjujeon/hyungju/data/aligned_volumes'
    (_, _, filenames) = next(os.walk(os.path.join(folder)))
    r = re.compile('.*tiff')
    filenames = list(filter(r.match, filenames))
    ch_list = []
    ants_image_list = [[]]*len(filenames)*2
    weights = np.repeat(1.0 / len(filenames), len(ch_list))

    for idx in range(len(filenames)):
        temp = ZImg(os.path.join(folder, filenames[idx]))
        img_data = temp.data[0][1,:,:,:].copy()
        temp = ants.from_numpy(np.moveaxis(img_data.astype('float32'), [0, 1], [-1, -2]))
        # temp = ants.image_read(os.path.join(folder, filenames[idx]))
        temp.set_spacing((10,10,100))
        # split_image = ants.split_channels(temp)
        ants_image_list[2*idx] = temp.resample_image((25,25,25))
        ants_image_list[2*idx+1] = ants.reflect_image(temp.resample_image((25,25,25)),0)

        image_ch = re.split("[_ .]",filenames[idx])[1:-1]
        for ch in image_ch:
            ch = ch.lower()
            if ch not in ch_list:
                ch_list.append(ch)
    weights = np.repeat(1.0 / len(ants_image_list), len(ch_list))



    # ants_template = build_template(image_list = ants_image_list)
    iterations = 30
    gradient_step = 0.2
    blending_weight = 0.5

    initial_name = os.path.join(folder, f'template_22.nii.gz')
    initial_template = ants.image_read(initial_name)
    initial_template.set_spacing((25, 25, 25))
    
    type_of_transform = "SyNAggro"
    # initial_template = ants_image_list[0] * 0
    # for i in range(len(ants_image_list)):
    #     temp = ants_image_list[i] * weights[0]
    #     temp = ants.resample_image_to_target(temp, initial_template)
    #     initial_template = initial_template + temp

    xavg = initial_template.clone()
    for i in range(1,iterations):
        for k in range(len(ants_image_list)):
            w1 = ants.registration(
                xavg, ants_image_list[k], type_of_transform=type_of_transform, verbose=False
            )
            if k == 0:
                wavg = ants.image_read(w1["fwdtransforms"][0]) * weights[0]
                xavgNew = w1["warpedmovout"] * weights[0]
            else:
                wavg = wavg + ants.image_read(w1["fwdtransforms"][0]) * weights[0]
                xavgNew = xavgNew + w1["warpedmovout"] * weights[0]
            fwdfn = os.path.join(folder, f'list_{k}_tform_0.nii.gz')
            shutil.copyfile(w1["fwdtransforms"][0], fwdfn)
            fwdfn = os.path.join(folder, f'list_{k}_tform_1.mat')
            shutil.copyfile(w1["fwdtransforms"][1], fwdfn)

        print(wavg.abs().mean())
        wscl = (-1.0) * gradient_step
        wavg = wavg * wscl
        wavgfn = mktemp(suffix=".nii.gz", dir=os.path.join(folder, 'temp'))
        ants.image_write(wavg, wavgfn)
        xavg = ants.apply_transforms(xavgNew, xavgNew, wavgfn)
        if blending_weight is not None:
            xavg = xavg * blending_weight + ants.iMath(xavg, "Sharpen") * (
                1.0 - blending_weight
            )
        os.remove(wavgfn)
        xavgfn = os.path.join(folder, f'template_{i}.nii.gz')
        ants.image_write(xavg, xavgfn)
