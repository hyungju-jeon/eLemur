import os

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '128'
import sys
import json
import glob
import copy
import pathlib
import traceback
import multiprocessing
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
from utils.brain_info import read_brain_info
from utils import shading_correction

logger = setup_logger()


def _callback(result):
    logger.info(f'finished {result}')


def _error_callback(e: BaseException):
    traceback.print_exception(type(e), e, e.__traceback__)
    raise e


def get_midline_average(mid_dict: dict, img_slice: int):
    mid_x = mid_dict['Regions'][-1]['ROI']['SliceROIs'][img_slice][0][0]['Points'][:, 0]
    res = sum(mid_x) / len(mid_x)
    return res


def get_flip_id(group_id: int, region_list=list):
    if group_id > 10:
        return group_id - 10
    else:
        if len(region_list) > 0:
            if group_id + 10 in region_list:
                return group_id + 10
            else:
                return group_id
        else:
            return group_id + 10


def get_slice_centroid(ra_dict: dict, slice_idx: int, x_axis: int, *, roi_id: int = None):
    x_pts = np.array([], dtype='f')

    for region_id, region in ra_dict['Regions'].items():
        if roi_id is not None:
            if region_id != roi_id:
                continue
        if region['ROI'] is not None:
            for img_slice, sliceROIs in region['ROI']['SliceROIs'].items():
                if img_slice != slice_idx:
                    continue
                for shape in sliceROIs:
                    for subShape in shape:
                        x_pts = np.concatenate((x_pts, subShape['Points'][:, x_axis]))
                        # return sum(x_pts) / len(x_pts)
    return (max(x_pts) + min(x_pts)) / 2


def get_region_side(ra_dict: dict, slice_idx: int, region_id: int, mid_x: int, x_axis: int):
    region_shape_ra = ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx]
    slice_midline = get_slice_centroid(ra_dict, slice_idx, x_axis)

    if len(region_shape_ra) > 2:
        region_midline = get_slice_centroid(ra_dict, slice_idx, x_axis, roi_id=region_id)
        if abs(mid_x - region_midline) > abs(mid_x - slice_midline):
            return mid_x > region_midline
        else:
            return mid_x > slice_midline
    else:
        return mid_x > get_slice_centroid(ra_dict, slice_idx, x_axis)


def get_group_status(stacked_label_filename: str, region_group_name: str):
    label_ZImg = ZImg(stacked_label_filename)
    label_volume = label_ZImg.data[0][0, :, :, :]

    fid = open(region_group_name, 'w')

    fid.write('[')
    newline_string = ''
    for i in range(np.shape(label_volume)[0]):
        label_slice = label_volume[i, :, :]
        region_list = np.unique(label_slice[label_slice > 0])
        fid.write(newline_string + '\n[[')
        region_string = ''
        for id in region_list:
            region_string = region_string + "%d, " % id
        region_string = region_string[:-2]
        fid.write(region_string + ']]')
        newline_string = ','
    fid.write('\n]')
    fid.close()


def manual_correct_single_image(slice_idx: int, image_folder: str, lemur_prefix: str, brain_info: dict = None):
    slice_folder_idx = slice_idx + 1
    tform_folder = os.path.join(image_folder, 'correction', f'{slice_folder_idx}')

    if brain_info is None:
        scene_idx = (slice_idx) % 4 + 1
        file_idx = slice_idx // 4 + 1
    else:
        scene_idx = int(brain_info['scene'][slice_idx])
        file_idx = int(brain_info['filename'][slice_idx][-6:-4])
    if scene_idx < 0:
        return

    pathlib.Path(os.path.join(image_folder, 'corrected')).mkdir(parents=True, exist_ok=True)
    res_filename = os.path.join(image_folder, 'corrected', f'{lemur_prefix}_{file_idx:02}_scene{scene_idx}_final.nim')
    if os.path.exists(res_filename):
        logger.info(f'{lemur_prefix}_{file_idx:02}_scene{scene_idx} already exist')
    else:
        # Read aligned image
        nim_filename = os.path.join(image_folder, f'{lemur_prefix}_{file_idx:02}_scene{scene_idx}_aligned.nim')
        logger.info(f'Correcting {lemur_prefix}_{file_idx:02}_scene{scene_idx}')
        if os.path.exists(nim_filename):
            img_ZImg = ZImg(nim_filename, region=ZImgRegion())
            img_data = img_ZImg.data[0][:, :, :, :].copy()
            img_info = ZImg.readImgInfos(nim_filename)
            img_info = img_info[0]
            des_width = img_info.width
            des_height = img_info.height
        else:
            return

        # Get label image
        label_name = os.path.join(tform_folder, 'image_label.mhd')
        if not os.path.exists(label_name):
            img_ZImg.save(res_filename)
            logger.info(f'image {slice_idx} done')
            return

        label_ZImg = ZImg(label_name)
        label_img = label_ZImg.data[0][0, 0, :, :].copy()
        region_list = np.unique(label_img)
        region_list = region_list[region_list != 0].astype(np.uint8)
        # Resize label image to raw resolution
        label_img = cv2.resize(label_img, dsize=(img_info.width, img_info.height),
                               interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # Get transform info
        corrected_img_data = np.zeros(img_data.shape, dtype=np.uint16)
        tform_name = os.path.join(tform_folder, 'manual_tforms.mat')
        tform_mat = scipy.io.loadmat(tform_name)
        tform_mat = tform_mat['tform_mat'][0]

        # For each unique region
        for region_id in region_list:
            #   Mask region using label
            mask = (label_img == region_id).astype(np.uint16)

            #   Rescale transform
            tform = tform_mat[region_id - 1].copy().astype(np.float64)
            tform[2, 0:2] = tform[2, 0:2] * 16
            #   Apply transform
            for ch in range(corrected_img_data.shape[0]):
                # masked_img = cv2.bitwise_and(img_data[ch,0,:,:],img_data[ch,0,:,:], mask)
                masked_img = img_data[ch, 0, :, :].copy()
                masked_img[mask == 0] = 0
                corrected_mask = cv2.warpAffine(mask, tform.T[0:2, :], dsize=(des_width, des_height),
                                                flags=cv2.INTER_NEAREST,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=0)
                corrected_region = cv2.warpAffine(masked_img, tform.T[0:2, :], dsize=(des_width, des_height),
                                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                                  borderValue=0)
                current_slice = corrected_img_data[ch, 0, :, :].copy()
                current_slice[corrected_mask > 0] = corrected_region[corrected_mask > 0]
                corrected_img_data[ch, 0, :, :] = current_slice


        img2 = ZImg(corrected_img_data, img_info)
        logger.info(f'saving {res_filename}')
        img2.save(res_filename)
        logger.info(f'image {slice_idx} done')


def pool_manual_correct_single_image(parameter_tuple: list):
    manual_correct_single_image(slice_idx=parameter_tuple[0], image_folder=parameter_tuple[1],
                                lemur_prefix=parameter_tuple[2],
                                brain_info=parameter_tuple[3])


def manual_correct_annotation(ra_filename: str, param_folder: str):
    read_ratio = 16
    scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
    # Read region annotation file
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_filename}')

    # Iterate all region
    for region_id, region in ra_dict['Regions'].items():
        if region['ROI'] is not None:
            for img_slice, sliceROIs in region['ROI']['SliceROIs'].items():
                logger.info(f'Correcting region {region_id} in slice {img_slice}')
                slice_folder_idx = img_slice + 1
                tform_folder = os.path.join(param_folder, f'{slice_folder_idx}')

                # Read image label file
                label_name = os.path.join(tform_folder, 'image_label.mhd')
                label_ZImg = ZImg(label_name)
                label_img = label_ZImg.data[0][0, 0, :, :].copy().astype(np.uint8)
                region_list = np.unique(label_img)
                region_list = region_list[region_list != 0].astype(np.uint8)

                for shape in sliceROIs:
                    # Iterate all shape
                    subShapes = []
                    for subShape in shape:
                        subShapes.append((subShape['Points'], subShape['Type'], subShape['IsAdd']))
                    compact_mask_ZImg, x_start, y_start = ZROIUtils.shapeToMask(subShapes)
                    compact_mask = compact_mask_ZImg.data[0][0, 0, :, :].copy()

                    y_max = min(label_img.shape[0], (y_start + compact_mask.shape[0]))
                    x_max = min(label_img.shape[1], (x_start + compact_mask.shape[1]))
                    compact_mask = compact_mask[:y_max - y_start, :x_max - x_start]

                    # Determine region-shape correspondence
                    compact_label = label_img[y_start:y_start + compact_mask.shape[0],
                                    x_start:x_start + compact_mask.shape[1]]
                    mode_array = scipy.stats.mode(compact_label[compact_mask == 1])[0]

                    if len(mode_array) == 0:
                        continue
                    else:
                        roi_id = mode_array[0]

                    # Load corresponding transform
                    tform_name = os.path.join(tform_folder, 'manual_tforms.mat')
                    tform_mat = scipy.io.loadmat(tform_name)
                    tform_mat = tform_mat['tform_mat'][0]
                    # Rescale transform
                    tform = tform_mat[roi_id - 1].copy().astype(np.float64)

                    # tform[2, 0:2] = tform[2, 0:2] * 16

                    # Apply transform
                    def transform_fun(input_coords: np.ndarray):
                        assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
                        res = np.concatenate((input_coords,
                                              np.ones(shape=(input_coords.shape[0], 1), dtype=np.float64)),
                                             axis=1)
                        # rotation back
                        M1 = np.array([[1., 0., 0.], [0, 1, 0], [0, 0, 1]])
                        res = M1 @ res.T
                        res = tform.T @ res
                        res = res.T[:, 0:2]
                        assert res.ndim == 2 and res.shape[1] == 2, res.shape
                        return res

                    for subShape in shape:
                        subShape['Points'] = transform_fun(subShape['Points'])

    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * read_ratio)
    result_filename = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset' \
                      '/Hotsauce_SMI99_VGluT2_NeuN/99_corrected_2d_reference.reganno'
    region_annotation.write_region_annotation_dict(ra_dict, result_filename)


if __name__ == "__main__":
    folder_lists = ['Fig_325AA/180918_Lemur-Fig_PV_TH_NeuN',
                   'Fig_325AA/180914_fig_SMI99_NeuN_VGlut2',
                   'Garlic_320CA/181023_Lemur-Garlic_SMI99_VGluT2_M2',
                   'Hotsauce_334A/181016_Lemur-Hotsauce_PV_TH_NeuN',
                   'Icecream_225BD/190221_icecream_PV_TH_NeuN',
                   'Icecream_225BD/20190218_icecream_SMI99_NeuN_VGlut2',
                   'Jellybean_289BD/20190813_jellybean_FOXP2_SMI32_NeuN',
                   'Jellybean_289BD/20190827_jellybean_vGluT2_SMI32_vGluT1']
    prefix_lists = ['Lemur-F_PV_TH_NeuN',
                   'Lemur-F_SMI99_NeuN_VGlut2',
                   'Lemur-G_SMI99_VGluT2_M2',
                   'Lemur-H_PV_TH_NeuN',
                   'Lemur-I_PV_TH_NeuN',
                   'Lemur-I_SMI99_VGluT2_NeuN',
                   'Lemur-J_FOXP2_SMI32_NeuN',
                   'Lemur-J_vGluT2_SMI32_vGluT1']

    for idx in [3]:
        lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')
        hyungju_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align'

        #result_list = 'Hotsauce_SMI99_VGluT2_NeuN'
        folder_list = folder_lists[idx]
        prefix_list = prefix_lists[idx]

        is_reference = False

        czi_folder = os.path.join(lemur_folder, folder_list)
        lemur_prefix = prefix_list
        brain_info_name = os.path.join(czi_folder, 'info.txt')
        brain_info = None
        if os.path.exists(brain_info_name):
            brain_info = read_brain_info(brain_info_name)

        image_folder = os.path.join(czi_folder, 'background_corrected', 'aligned')

        param_set = [(idx, image_folder, lemur_prefix, brain_info) for idx in range(200)]
        # pool_manual_correct_single_image(param_set[160])
        # sys.exit(1)
        with multiprocessing.Pool(8) as pool:
            pool.map_async(pool_manual_correct_single_image, param_set, chunksize=1, callback=None,
                           error_callback=_error_callback).wait()
        # sys.exit(1)

        # Correct annotation if dataset is reference
        if False:
            if is_reference:
                manual_correct_annotation(os.path.join(hyungju_folder,
                                                'all-dataset/Hotsauce_SMI99_VGluT2_NeuN/00_stacked_annotation_with_layer.reganno'),
                                          os.path.join(image_folder, 'correction'))
        # sys.exit(1)

        if True:
            # (_, _, file_list) = next(os.walk(os.path.join(image_folder, 'corrected')))
            (_, _, file_list) = next(os.walk(image_folder))
            finalfile_list = [fn for fn in file_list if '_final.nim' in fn]

            combined_filename = os.path.join(image_folder, 'corrected', f'{lemur_prefix}_all.nim')
            if os.path.exists(combined_filename):
                logger.info(f'image {combined_filename} done')
            else:
                if brain_info is None:
                    scene_list = [x % 4 + 1 for x in range(len(file_list))]
                    file_list = [x // 4 + 1 for x in range(len(file_list))]
                else:
                    scene_list = [int(x) for x in brain_info['scene']]
                    file_list = [int(x[-6:-4]) for x in brain_info['filename']]

                aligned_filenames = [os.path.join(image_folder, 'corrected',
                                                  f'{lemur_prefix}_{file_list[img_idx]:02}_scene'
                                                  f'{scene_list[img_idx]}_final.nim') for img_idx in
                                     range(len(file_list))]
                # aligned_filenames = [os.path.join(image_folder,
                #                                   f'{lemur_prefix}_{file_list[img_idx]:02}_scene'
                #                                   f'{scene_list[img_idx]}_aligned.nim') for img_idx in
                #                      range(len(file_list))]
                # aligned_filenames = [os.path.join(image_folder, 'corrected', file_list[img_idx]) for img_idx in
                #                      range(len(file_list))]
                imgMerge = ZImgMerge()
                imgSubBlocks = []
                for idx, fn in enumerate(aligned_filenames):
                    if not os.path.exists(fn):
                        continue
                    imgSubBlocks.append(ZImgTileSubBlock(ZImgSource(fn)))
                    imgMerge.addImg(imgSubBlocks[-1], (0, 0, idx, 0, 0), pathlib.PurePath(fn).name)
                imgMerge.resolveLocations()
                imgMerge.save(combined_filename)
                logger.info(f'image {combined_filename} done')
