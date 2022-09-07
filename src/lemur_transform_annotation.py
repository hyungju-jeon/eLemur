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
import logging

from zimg import *
from utils import io
from utils import img_util
from utils import nim_roi
from utils import region_annotation
from utils.logger import setup_logger
from utils import shading_correction
from utils.brain_info import read_brain_info
from skimage.segmentation import watershed, expand_labels
import skimage
from tempfile import mktemp

logger = setup_logger()


def _callback(result):
    logger.info(f'finished {result}')


def composite_transformation(tx_a:str, tx_b: str):
    param_a = ants.read_transform(tx_a).parameters
    param_b = ants.read_transform(tx_b).parameters

    mat_A = np.array([[param_a[0], param_a[1], param_a[4]], [param_a[2], param_a[3], param_a[5]],[0,0,1]])
    mat_B = np.array([[param_b[0], param_b[1], param_b[4]], [param_b[2], param_b[3], param_b[5]],[0,0,1]])

    mat_C = np.matmul(mat_A,mat_B)
    txfn = mktemp(suffix=".mat")

    final_param = [mat_C[0,0], mat_C[0,1],mat_C[1,0],mat_C[1,1],mat_C[0,2],mat_C[1,2]]
    new_tx = ants.create_ants_transform(transform_type='AffineTransform',precision='float', dimension=2,parameters=final_param)
    ants.write_transform(new_tx, txfn)

    return txfn


def _error_callback(e: BaseException):
    traceback.print_exception(type(e), e, e.__traceback__)
    raise e


def get_midline_average(mid_dict: dict, img_slice: int):
    mid_x = mid_dict['Regions'][-1]['ROI']['SliceROIs'][img_slice][0][0]['Points'][:,0]
    res = sum(mid_x)/len(mid_x)
    return res


def get_flip_id(group_id:int, region_list = list):
     if group_id > 10:
         return group_id - 10
     else:
         if len(region_list)>0 :
             if group_id+10 in region_list:
                 return group_id+10
             else:
                 return group_id
         else:
             return group_id + 10


def get_slice_centroid(ra_dict: dict, slice_idx: int, x_axis: int, *, roi_id: int = None):
    x_pts = np.array([],dtype='f')

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
                            x_pts = np.concatenate((x_pts,subShape['Points'][:,x_axis]))
    # return sum(x_pts) / len(x_pts)
    return (max(x_pts) + min(x_pts))/2


def get_region_side(ra_dict: dict, slice_idx: int, region_id: int, mid_x: int, x_axis: int):
    region_shape_ra = ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx]
    slice_midline = get_slice_centroid(ra_dict, slice_idx, x_axis)

    if len(region_shape_ra) > 2:
        region_midline = get_slice_centroid(ra_dict, slice_idx, x_axis, roi_id = region_id)
        if abs(mid_x-region_midline) > abs(mid_x-slice_midline):
            return mid_x > region_midline
        else:
            return mid_x > slice_midline
    else:
        return mid_x > get_slice_centroid(ra_dict, slice_idx, x_axis)


def get_blockface_status(stacked_label_filename:str, blockface_group_name:str, blockface_split_name:str):
    label_ZImg = ZImg(stacked_label_filename)
    label_volume = label_ZImg.data[0][0,:,:,:]

    fid = open(blockface_group_name, 'w')

    fid.write('[')
    newline_string = ''
    for i in range(np.shape(label_volume)[0]):
        label_slice = label_volume[i,:,:]
        region_list = np.unique(label_slice[label_slice>0])
        fid.write(newline_string+'\n[[')
        region_string= ''
        for id in region_list:
            region_string = region_string + "%d, "%id
        region_string = region_string[:-2]
        fid.write(region_string+']]')
        newline_string = ','
    fid.write('\n]')
    fid.close()

    fid = open(blockface_split_name, 'w')

    fid.write('[')
    newline_string = ''
    for i in range(np.shape(label_volume)[0]):
        fid.write(newline_string+'\n[')
        fid.write('-1]')
        newline_string = ','
    fid.write('\n]')
    fid.close()


def get_group_status(stacked_label_filename:str, region_group_name:str):
    label_ZImg = ZImg(stacked_label_filename)
    label_volume = label_ZImg.data[0][0,:,:,:]

    fid = open(region_group_name, 'w')

    fid.write('[')
    newline_string = ''
    for i in range(np.shape(label_volume)[0]):
        label_slice = label_volume[i,:,:]
        region_list = np.unique(label_slice[label_slice>0])
        fid.write(newline_string+'\n[[')
        region_string= ''
        for id in region_list:
            region_string = region_string + "%d, "%id
        region_string = region_string[:-2]
        fid.write(region_string+']]')
        newline_string = ','
    fid.write('\n]')
    fid.close()


def export_grouped_label_img(img_filename: str, ra_filename: str,
                             region_group: list, group_split: list,
                             result_filename: str,
                             *, midline_filename: str = None, midline_x: float = None, downsample_ratio: int = 16,
                             group_flip: str = None):
    read_ratio = downsample_ratio
    scale_down = 1.0 / read_ratio  # otherwise the mask will be too big

    infoList = ZImg.readImgInfos(img_filename)
    assert len(infoList) == 1 and infoList[0].numTimes == 1
    img_info = infoList[0]
    logger.info(f'image {infoList[0]}')

    img_rotate = True if img_info.height > img_info.width else False

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_filename}')
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    logger.info(f'finish reading masks from {ra_filename}')

    region_id_set = set()
    for region_id, slice_rois in region_to_masks.items():
        if region_id < 0:
            continue
        for img_slice, maskps in slice_rois.items():
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                region_id_set.add(region_id)

    if midline_filename is not None:
        ra_dict2 = region_annotation.read_region_annotation(midline_filename)
        ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict2, lambda coords: coords * scale_down)
        logger.info(f'finish reading {midline_filename}')
        midline_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict2)
        logger.info(f'finish reading masks from {midline_filename}')

    if img_rotate:
        annotation_mask = np.zeros(shape=(img_info.depth,
                                          int(math.ceil(img_info.height)),
                                          int(math.ceil(img_info.width))),
                                   dtype=np.uint8)
    else:
        annotation_mask = np.zeros(shape=(img_info.depth,
                                          int(math.ceil(img_info.height * scale_down)),
                                          int(math.ceil(img_info.width * scale_down))),
                                   dtype=np.uint8)

    for region_id, slice_rois in region_to_masks.items():
        if region_id not in region_id_set:
            continue
        for slice_idx, maskps in slice_rois.items():
            # if(slice_idx != 132):
            #     continue
            logger.info(f'running {slice_idx}')
            mapped_region_id = 0
            for group_idx, group in enumerate(region_group[slice_idx], start=1):
                if region_id in group:
                    mapped_region_id = group_idx
                    break
                if ra_dict['ID_To_ParentID'][region_id] in group:
                    mapped_region_id = group_idx
                    break

            if mapped_region_id == 0:
                mapped_region_id = len(region_group[slice_idx]) + 1

            need_split = mapped_region_id in group_split[slice_idx]

            for compact_mask, x_start, y_start, spline in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                mask = np.zeros(shape=(annotation_mask.shape[-2], annotation_mask.shape[-1]), dtype=np.bool)
                # if img_rotate:
                #     compact_mask = np.moveaxis(compact_mask, -1, -2)
                #     temp = x_start
                #     x_start = y_start
                #     y_start = temp
                y_max = min(mask.shape[0], (y_start + compact_mask.shape[0]))
                x_max = min(mask.shape[1], (x_start + compact_mask.shape[1]))
                compact_mask = compact_mask[:y_max-y_start,:x_max-x_start]
                mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = compact_mask

                if need_split:
                    if midline_filename is not None:
                        midline_slice_rois = midline_to_masks[-1]
                        assert slice_idx in midline_slice_rois, slice_idx
                        midline_maskps = midline_slice_rois[slice_idx]
                        midline_mask = np.zeros(shape=mask.shape, dtype=np.bool)
                        for midline_compact_mask, midline_x_start, midline_y_start, _ in midline_maskps:
                            if midline_compact_mask.sum() == 0:
                                continue
                            assert midline_x_start >= 0 and midline_y_start >= 0, (
                                midline_x_start, midline_y_start, midline_compact_mask.shape)
                            midline_mask[midline_y_start:midline_y_start + midline_compact_mask.shape[0],
                                         midline_x_start:midline_x_start + midline_compact_mask.shape[1]] = \
                                midline_compact_mask
                        mask_centroid = scipy.ndimage.measurements.center_of_mass(midline_mask)
                        if x_start + compact_mask.shape[1] / 2 > mask_centroid[-1]:  # right side
                            annotation_mask[slice_idx][mask] = mapped_region_id + 10
                        else:
                            annotation_mask[slice_idx][mask] = mapped_region_id
                    else:
                        if midline_x is not None:
                            if np.mean(spline[0]['Points'][:,0]) > midline_x * scale_down:  # right side
                                annotation_mask[slice_idx][mask] = mapped_region_id + 10
                            else:
                                annotation_mask[slice_idx][mask] = mapped_region_id
                        else:
                            if img_rotate:
                                mid_x = np.mean(spline[0]['Points'][:,1])
                                on_right_hemisphere = get_region_side(ra_dict, slice_idx, region_id, mid_x, 1)
                                if  on_right_hemisphere:  # right side
                                    annotation_mask[slice_idx][mask] = mapped_region_id + 10
                                else:
                                    annotation_mask[slice_idx][mask] = mapped_region_id
                            else:
                                mid_x = np.mean(spline[0]['Points'][:,0])
                                on_right_hemisphere = get_region_side(ra_dict, slice_idx, region_id, mid_x, 0)
                                if on_right_hemisphere:  # right side
                                    annotation_mask[slice_idx][mask] = mapped_region_id + 10
                                else:
                                    annotation_mask[slice_idx][mask] = mapped_region_id
                else:
                    annotation_mask[slice_idx][mask] = mapped_region_id

    img_util.write_img(result_filename, annotation_mask)


def apply_inverse_transform_annotation(ra_filename: str, region_group: list, group_split: list,
                                      group_flip: list, group_error: list, result_filename: str, tform_folder: str,
                                     *, downsample_ratio: int = 16):
    # Read annotation and get dict (nx2 array)
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    logger.info(f'finish reading {ra_filename}')

    # Iterate over each region > slice > shape
    for region_id, region in ra_dict['Regions'].items():
        if region['ROI'] is not None:
            # if region_id != -1:
                # continue

            for img_slice, sliceROIs in region['ROI']['SliceROIs'].items():
                #assert img_slice_ % 2 == 0, img_slice_
                #img_slice = int(img_slice_ / 2)

                mapped_region_id = 0
                # Get the group ID for the region
                for group_idx, group in enumerate(region_group[img_slice], start=1):
                    if region_id in group or region['ParentID'] in group:
                        mapped_region_id = group_idx
                        break

                # If the region is not mapped, ignore
                if mapped_region_id == 0:
                    continue

                # Check the region in the slice need cutting
                need_split = mapped_region_id in group_split[img_slice]

                for shape in sliceROIs:
                    for subShape in shape:
                        shape_id = mapped_region_id
                        if(need_split):
                            shape_id = mapped_region_id + 10

                        # Get annotation file name
                        deform_filename = os.path.join(tform_folder, f"{img_slice+1}", f"{img_slice+1}_{shape_id}_fwd.h5")
                        if os.path.exists(deform_filename):
                            # Read affine elastix transform
                            def transform_fun(input_coords: np.ndarray):
                                assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape

                                res = pd.DataFrame(data=input_coords.copy(), columns=['x', 'y'])
                                res = res/downsample_ratio
                                res = ants.apply_transforms_to_points(2, res, deform_filename)
                                res = res * downsample_ratio
                                res = res.to_numpy()

                                assert res.ndim == 2 and res.shape[1] == 2, res.shape
                                return res

                            # Apply transform
                            subShape['Points'] = transform_fun(subShape['Points'])
                            print(deform_filename)
                        else:
                            subShape = None

                        if((shape_id in group_flip[img_slice])):
                            shape_id = mapped_region_id

                        # Get annotation file name
                        tform_filename = os.path.join(tform_folder, f"{img_slice+1}", f"{img_slice+1}_{shape_id}-trns.txt")
                        if os.path.exists(tform_filename):
                            # Read affine elastix transform
                            affine_tform = get_elastix_transform(tform_filename)
                            affine_tform = np.linalg.inv(affine_tform)
                            def transform_fun(input_coords: np.ndarray):
                                assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
                                res = input_coords.copy()
                                res = res @ affine_tform[0:2, 0:2]
                                res[:,0] += affine_tform[2,0]
                                res[:,1] += affine_tform[2,1]
                                assert res.ndim == 2 and res.shape[1] == 2, res.shape
                                return res

                            # Apply transform
                            subShape['Points'] = transform_fun(subShape['Points'])
                        else:
                            subShape = None

                        need_flip = (shape_id in group_flip[img_slice]) ^ (shape_id in group_error[img_slice])
                        # Define flip function (TODO : HOW DO I GET IMAGE WIDTH)
                        def transform_fun(input_coords: np.ndarray):
                            assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
                            res = input_coords.copy()
                            res[:, 0] = 28080 - res[:, 0]
                            assert res.ndim == 2 and res.shape[1] == 2, res.shape
                            return res
                        # Apply flip function (TODO : HOW DO I GET IMAGE WIDTH)
                        if(need_flip):
                            subShape['Points'] = transform_fun(subShape['Points'])



    # Write Region Annotation
    region_annotation.write_region_annotation_dict(ra_dict, result_filename)


def run_rigid_transform_slice(reflabel_name: str, movlabel_name: str, result_filename: str, tform_folder: str, is_scaling: bool, group_flip:list = None, group_error:list = None):
    img = ZImg(movlabel_name)
    mov_volume = img.data[0][0,:,:,:].copy()
    mov_volume[mov_volume<0] = 0

    infoList = ZImg.readImgInfos(movlabel_name)
    assert len(infoList) == 1 and infoList[0].numTimes == 1
    img_info = infoList[0]

    img_rotate = True if img_info.height > img_info.width else False
    if not img_rotate:
        mov_volume = np.moveaxis(mov_volume,-2,-1)

    fix_img = ZImg(reflabel_name)
    fix_volume = fix_img.data[0][0,:,:,:].copy()
    fix_volume = ants.from_numpy(np.moveaxis(fix_volume, [0,1], [-1,-2]))

    result_volume = np.zeros(shape = (fix_volume.shape[2],fix_volume.shape[0],fix_volume.shape[1]), dtype=fix_volume.dtype)

    tform_type = 'Similarity' if is_scaling else 'Rigid'

    for slice_idx in range(fix_volume.shape[2]):
    # for slice_idx in range(13,14):
        expanded_fix_slice = expand_labels(fix_volume.slice_image(2, slice_idx).numpy(), distance=30)
        logger.info(f'running {slice_idx}')
        slice_folder_idx = slice_idx + 1
        mov_folder = os.path.join(tform_folder, str(slice_folder_idx))
        pathlib.Path(mov_folder).mkdir(parents=True, exist_ok=True)

        result_slice = np.zeros(shape = (fix_volume.shape[0], fix_volume.shape[1]), dtype=fix_volume.dtype)
        for group_id in np.unique(fix_volume.slice_image(2, slice_idx).numpy()).astype('uint8'):
            if group_id == 0:
                continue

            # Check whether the image needs to be flipped
            if group_flip != None :
                is_flip = group_id in group_flip[slice_idx]
            else:
                is_flip = False
            if group_error != None:
                is_error = group_id in group_error[slice_idx]
            else:
                is_error = False

            fix_group_id = group_id
            if is_flip:
                fix_group_id = get_flip_id(group_id, np.unique(fix_volume.slice_image(2, slice_idx).numpy()).astype('uint8'))

            mov_name = os.path.join(mov_folder, f'{slice_folder_idx}_{fix_group_id}.mhd')
            if os.path.exists(mov_name):
                mov = ants.image_read(mov_name)

                mov_result = mov.numpy().astype(result_volume.dtype)
                result_slice[mov_result > 0] = mov_result[mov_result > 0]*fix_group_id;

            else:
                mov_slice = mov_volume[slice_idx,:,:]
                fix_group_id = group_id
                if is_flip:
                    mov_slice = np.flipud(mov_slice)
                    fix_group_id = get_flip_id(group_id, np.unique(fix_volume.slice_image(2, slice_idx).numpy()).astype('uint8'))
                if is_error:
                    if slice_idx > 167:
                        mov_slice = (np.fliplr(mov_slice))
                    else:
                        mov_slice = (np.flipud(mov_slice))

                mov_slice = ants.from_numpy(mov_slice)

                mov = ants.get_mask(mov_slice, low_thresh = group_id, high_thresh = group_id, cleanup=0)
                mov = ants.morphology(mov, operation='close', radius=20, mtype='binary', shape='ball')
                mov = ants.iMath(mov, 'FillHoles')

                if np.sum(mov.numpy()) < 10:
                    continue

                fix = fix_volume.slice_image(2, slice_idx)
                fix = ants.get_mask(fix, low_thresh = fix_group_id, high_thresh = fix_group_id, cleanup=0)
                # if slice_idx in [161,162] and group_id in [1,11]:
                #     fix = fix_volume.slice_image(2, slice_idx-1)
                #     fix = ants.get_mask(fix, low_thresh = fix_group_id, high_thresh = fix_group_id, cleanup=0)
                if is_error:
                    if slice_idx < 168 and slice_idx > 80:
                        ang = skimage.measure.regionprops(skimage.measure.label(mov.numpy()>0))[0].orientation * 2.0
                        param = [np.cos(ang), -np.sin(ang), np.sin(ang), np.cos(ang), 0, 0]
                        cent =  scipy.ndimage.measurements.center_of_mass(mov.numpy())
                        param[-2] = cent[0]-(param[0] * cent[0] + param[1] * cent[1])
                        param[-1] = cent[1]-(param[2] * cent[0] + param[3] * cent[1])
                        txfn =mktemp(suffix=".mat")
                        new_tx= ants.create_ants_transform(transform_type='AffineTransform',precision='float', dimension=2,parameters=param)

                        ants.write_transform(new_tx, txfn)
                        mov2 = ants.apply_transforms(fix,mov,txfn)
                        mytx = ants.registration(fixed=fix , moving=mov2, type_of_transform=tform_type, aff_metric='meansquares', write_composite_transform=False, verbose=False, grad_step = 0.5)
                        final_tx = composite_transformation(txfn, mytx['fwdtransforms'][0])
                        mytx['fwdtransforms'][0] = final_tx
                    else:
                        txfile = ants.affine_initializer( fix, mov, use_principal_axis=True,local_search_iterations=10, search_factor=10, radian_fraction=0.5)
                        mytx = ants.registration(fixed=fix , moving=mov, type_of_transform=tform_type, aff_metric='meansquares',
                         write_composite_transform=False, verbose=False, grad_step = 0.5,initial_transform= txfile)
                else:
                    mytx = ants.registration(fixed=fix , moving=mov, type_of_transform=tform_type, aff_metric='meansquares',
                                         write_composite_transform=False, verbose=False, grad_step = 0.1)
                tform_name = os.path.join(mov_folder, f'{slice_folder_idx}_{fix_group_id}.mat')
                shutil.copyfile(mytx['fwdtransforms'][0], tform_name)

                diff_const = 100
                mov = ants.from_numpy(ants.get_mask(mov, low_thresh = 1,cleanup=0).numpy()*diff_const*2)

                fix = fix_volume.slice_image(2, slice_idx)
                fix = ants.from_numpy(ants.get_mask(fix, low_thresh = fix_group_id, high_thresh = fix_group_id, cleanup=0).numpy()*diff_const
                                      + (expanded_fix_slice == fix_group_id)*diff_const)

                mytx = ants.registration(fixed=fix , moving=mov, initial_transform=tform_name, type_of_transform=tform_type,
                                         aff_metric='meansquares', write_composite_transform=False, verbose=False, grad_step = 0.05)
                shutil.copyfile(mytx['fwdtransforms'][0], tform_name)


                mov_result = ants.get_mask(ants.apply_transforms(fix, mov, tform_name, interpolator='nearestNeighbor'),low_thresh = diff_const, cleanup=0)
                mov_result.to_file(mov_name)

                mov_result = mov_result.numpy().astype(result_volume.dtype)
                result_slice[mov_result > 0] = mov_result[mov_result > 0]*fix_group_id;

            result_volume[slice_idx,:,:] = result_slice

    img_util.write_img(result_filename, np.moveaxis(result_volume, -1, -2))


def export_matching_reference(reflabel_name: str, movlabel_name: str, result_filename: str, result_folder: str,
                              reference_group: list, region_group: list, group_split: list):
    img = ZImg(movlabel_name)
    fix_volume = img.data[0][0,:,:,:]
    fix_volume[fix_volume<0]=0

    infoList = ZImg.readImgInfos(movlabel_name)
    assert len(infoList) == 1 and infoList[0].numTimes == 1
    img_info = infoList[0]

    img_rotate = True if img_info.height > img_info.width else False
    if not img_rotate:
        fix_volume = np.moveaxis(fix_volume,-2,-1)

    ants_mov_volume = ants.image_read(reflabel_name).astype('uint32')
    ants_fix_volume = ants.from_numpy(np.moveaxis(fix_volume.astype('uint32'),0,-1))

    # Get starting index difference
    reference_start = np.nonzero([len(fn) for fn in reference_group])[0][0]
    region_start = np.nonzero([len(fn) for fn in region_group])[0][0]

    # Transformation
    midslice_idx = int(np.round(np.shape(fix_volume)[0]/2))
    parameters = []
    for idx in range(80,90):
        ants_mov_slice = ants_mov_volume.slice_image(2, idx)
        ants_fix_slice = ants_fix_volume.slice_image(2, idx)

        mov = ants.get_mask(ants_mov_slice, low_thresh = 1,  cleanup=0)
        mov = ants.morphology(mov, operation='close', radius=10, mtype='binary', shape='ball')
        mov = ants.iMath(mov, 'FillHoles')

        fix = ants.get_mask(ants_fix_slice, low_thresh = 1,  cleanup=0)
        fix = ants.morphology(fix, operation='close', radius=10, mtype='binary', shape='ball')
        fix = ants.iMath(fix, 'FillHoles')

        # mytx = ants.registration(fixed=fix , moving=mov, type_of_transform='Similarity', aff_metric='meansquares',
        #                          write_composite_transform=False, verbose=False, grad_step = 0.1)
        mytx = ants.registration(fixed=fix , moving=mov, type_of_transform='Affine', aff_metric='meansquares',
                                 restrict_deformation='1x0x0x1x1x1',
                                 write_composite_transform=False, verbose=False, grad_step = 0.1)


        tx = ants.read_transform(mytx['fwdtransforms'][0])
        parameters.append(tx.parameters)

    average_scale = np.mean(parameters, axis = 0)
    # average_scale = average_scale[0]
    parameter_scale = [average_scale[0], 0., 0., 0., average_scale[3], 0., 0., 0., 1.,
                       (1-average_scale[0]) * mov.shape[0]/2, (1-average_scale[3]) * mov.shape[1]/2, 0.]
    new_tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3, parameters=parameter_scale)
    tform_name = os.path.join(result_folder, 'reference_match.mat')
    ants.write_transform(new_tx, tform_name)

    ants_matched_ref = ants.apply_transforms(fixed=ants_mov_volume, moving=ants_mov_volume, transformlist=tform_name, interpolator='nearestNeighbor')

    # Group matching
    group_matched_ref = np.zeros((ants_matched_ref.shape[0],ants_matched_ref.shape[1], len(reference_group)), dtype='uint8')

    for idx in range(len(reference_group)):
        logger.info(f'Running slice {idx}')
        slice_idx = idx - reference_start + region_start
        group_slice_idx = idx
        if slice_idx < 0:
            slice_idx = 0
            group_slice_idx = reference_start
        if slice_idx > 179:
            slice_idx = 179
            group_slice_idx = 179

        new_slice = group_matched_ref[:,:,idx]
        curr_slice = ants_matched_ref.slice_image(2, slice_idx).numpy()

        region_list = np.unique(curr_slice[curr_slice>0])
        for region_id in region_list:
            for group_idx, group in enumerate(reference_group[group_slice_idx], start=1):
                if region_id in group or region_id-10 in group:
                    if group_idx in group_split[group_slice_idx]:
                        temp_slice = np.zeros(shape=new_slice.shape, dtype = new_slice.dtype)
                        temp_slice[curr_slice == region_id] = group_idx if region_id < 10 else group_idx+10
                        if region_id < 10 and region_id+10 not in group:
                            mid_x = int(temp_slice.shape[0]/2)
                            temp_slice[mid_x:,:] += 10
                            temp_slice[new_slice == 10] = 0

                        new_slice[curr_slice == region_id] = temp_slice[curr_slice == region_id]
                    else:
                        new_slice[curr_slice == region_id] = group_idx
        group_matched_ref[:,:,idx] = new_slice

    img_util.write_img(result_filename, np.moveaxis(group_matched_ref,[-1,-2],[0,-2]))


def apply_rigid_transform_annotation(ra_filename: str, region_group: list, group_split: list, result_filename: str, tform_folder: str,
                                     *, img_filename: str = None, group_flip: list = None, group_error: list = None,
                                     midline_filename: str = None, midline_x: float = None, downsample_ratio: int = 16):
    # Read annotation and get dict (nx2 array)
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    logger.info(f'finish reading {ra_filename}')
    # Read midline and get dict (nx2 array)
    if midline_filename != None:
        mid_dict = region_annotation.read_region_annotation(midline_filename)
        logger.info(f'finish reading {midline_filename}')

    # Iterate over each region > slice > shape
    mid_slice = {}
    for region_id_, region in ra_dict['Regions'].items():
        if region['ROI'] is not None:
            if region_id_ == 4775:
                region['ParentID'] = 1097
            if region_id_ == -1:
                region_id = 315
            else:
                region_id = region_id_

            for img_slice, sliceROIs in region['ROI']['SliceROIs'].items():
                if img_slice not in mid_slice.keys():
                    mid_slice[img_slice] = get_slice_centroid(ra_dict, img_slice, 0)
                #assert img_slice_ % 2 == 0, img_slice_
                #img_slice = int(img_slice_ / 2)

                mapped_region_id = 0
                # Get the group ID for the region
                for group_idx, group in enumerate(region_group[img_slice], start=1):
                    if region_id in group:
                        mapped_region_id = group_idx
                        break
                    if region['ParentID'] in group:
                        mapped_region_id = group_idx
                        break
                # If the region is not mapped, ignore
                if mapped_region_id == 0:
                    continue

                # Check the region in the slice need cutting
                need_split = mapped_region_id in group_split[img_slice]

                for shape in sliceROIs:
                    for subShape in shape:
                        shape_id = mapped_region_id

                        if(need_split):
                            # Get average midline and check whehter the shape is on left/right hemisphere
                            mid_x = np.mean(subShape['Points'][:,0])
                            if midline_filename != None:
                                if(mid_x > get_midline_average(mid_dict, img_slice)):
                                    shape_id = mapped_region_id + 10
                            elif midline_x != None:
                                if(mid_x > midline_x):
                                    shape_id = mapped_region_id + 10
                            else:
                                if(mid_x > mid_slice[img_slice]):
                                    shape_id = mapped_region_id + 10

                        # Check whether the image needs to be flipped
                        if group_flip != None and group_error != None:
                            is_flip = shape_id in group_flip[img_slice]
                            is_error = shape_id in group_error[img_slice]
                        else:
                            is_flip = False
                            is_error = False

                        # Define flip function (TODO : HOW DO I GET IMAGE WIDTH)
                        def flip_lr(input_coords: np.ndarray):
                            assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
                            res = input_coords.copy()
                            res[:, 0] = 28080 - res[:, 0]
                            assert res.ndim == 2 and res.shape[1] == 2, res.shape
                            return res

                        # Apply flip function (TODO : HOW DO I GET IMAGE WIDTH)
                        if(is_flip):
                            if need_split:
                                shape_id = get_flip_id(shape_id, [])
                            subShape['Points'] = flip_lr(subShape['Points'])
                        if(is_error):
                            subShape['Points'] = flip_ud(subShape['Points'])

                        # Get annotation file name
                        elastix_tform = os.path.join(tform_folder, f"{img_slice+1}", f"{img_slice+1}_{shape_id}-trns.txt")
                        antspy_tform = os.path.join(tform_folder, f"{img_slice+1}", f"{img_slice+1}_{shape_id}.mat")

                        if os.path.exists(elastix_tform):
                            # Read elastix affine elastix transform
                            affine_tform = get_elastix_transform(elastix_tform)
                            def transform_fun(input_coords: np.ndarray):
                                assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
                                res = input_coords.copy()
                                res = res @ affine_tform[0:2, 0:2]
                                res[:,0] += affine_tform[2,0]
                                res[:,1] += affine_tform[2,1]
                                assert res.ndim == 2 and res.shape[1] == 2, res.shape
                                return res

                            # Apply transform
                            subShape['Points'] = transform_fun(subShape['Points'])
                        elif os.path.exists(antspy_tform):
                            def transform_fun(input_coords: np.ndarray):
                                assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
                                # res = pd.DataFrame(data=input_coords[:,::-1].copy(), columns=['x', 'y'])
                                res = pd.DataFrame(data=input_coords.copy(), columns=['x', 'y'])
                                res = res/downsample_ratio
                                res = ants.apply_transforms_to_points(2, res, antspy_tform, whichtoinvert=[True])
                                res = res * downsample_ratio
                                res = res.to_numpy()

                                assert res.ndim == 2 and res.shape[1] == 2, res.shape
                                return res

                            # Apply transform
                            subShape['Points'] = transform_fun(subShape['Points'])
                        else:
                            subShape = None


    # Write Region Annotation
    region_annotation.write_region_annotation_dict(ra_dict, result_filename)

    img_util.write_img(result_filename, combined_volume)


def apply_rigid_transform_image(result_filename: str, tform_folder: str, reflabel_name: str, movlabel_name: str,
                                czi_folder : str, lemur_prefix : str, *, group_split: list = None, group_flip: list = None, group_error: list = None, czi_filename : str = None, run_shading_correction: bool = False, brain_info : dict = None):
    mask_img = ZImg(movlabel_name)
    mask_volume = mask_img.data[0][0,:,:,:]
    # mask_volume[mask_volume<0] = 0

    fix_img = ZImg(reflabel_name)
    fix_volume = fix_img.data[0][0,:,:,:]
    fix_volume = np.moveaxis(fix_volume,-2,-1)
    # fix_volume[fix_volume<0] = 0

    infoList = ZImg.readImgInfos(movlabel_name)
    assert len(infoList) == 1 and infoList[0].numTimes == 1
    img_info = infoList[0]

    img_rotate = True if img_info.height > img_info.width else False
    if not img_rotate:
        mask_volume = np.moveaxis(mask_volume,-2,-1)

    combined_slice = []
    if czi_filename is not None:
        img_volume = ZImg(czi_filename, region=ZImgRegion(), xRatio=16, yRatio=16)

    for slice_idx in range(fix_volume.shape[0]):
    # for slice_idx in range(160,170):
        logger.info(f'running {slice_idx}')
        slice_folder_idx = slice_idx + 1
        slice_folder = os.path.join(tform_folder, str(slice_folder_idx))


        if brain_info is None:
            scene_idx = (slice_idx) % 4
            file_idx = math.ceil(slice_folder_idx/4)
        else:
            scene_idx = int(brain_info['scene'][slice_idx])-1
            file_idx = slice_idx
        if scene_idx < 0:
            img = np.zeros((5, fix_volume.shape[1], fix_volume.shape[2]), 'float32')
            for ch in range(0,img.shape[0]):
                tformed_slice[ch] = ants.from_numpy(img[ch,:,:])
            combined_slice.append(ants.merge_channels(tformed_slice).numpy().astype('uint16'))
            continue

        # Get CZI histology image
        if czi_filename is None:
            if brain_info is None:
                czi_name = os.path.join(czi_folder, f'{lemur_prefix}_{file_idx:02}.czi')
            else:
                czi_name = brain_info['filename'][slice_idx]
            if os.path.exists(czi_name):
                if run_shading_correction:
                    correct_img, imgInfo = shading_correction.correct_shading(czi_name, scene=scene_idx)
                    img = img_util.imresize(correct_img, des_depth=1,
                                            des_height = int(np.ceil(correct_img.shape[-2]/16)),
                                            des_width = int(np.ceil(correct_img.shape[-1]/16)))
                    img = img[:,0,:,:]
                else:
                    imgInfo = ZImg(czi_name, region=ZImgRegion(), scene=scene_idx, xRatio=16, yRatio=16)
                    img = imgInfo.data[0][:,0,:,:]
                if not img_rotate:
                    img = np.moveaxis(img,-2,-1)
                x_pad = mask_volume.shape[1]-img.shape[1]
                y_pad = mask_volume.shape[2]-img.shape[2]

                if x_pad > 0:
                    img = np.pad(img, ((0,0), (0, x_pad), (0, 0)))
                else:
                    img = img[:,:x_pad,:]

                if y_pad > 0:
                    img = np.pad(img, ((0,0), (0, 0), (0, y_pad)))
                else:
                    img = img[:,:,:y_pad]
            else:
                tformed_slice = [[]]*5
                img = np.zeros((5, fix_volume.shape[1], fix_volume.shape[2]), 'float32')
                for ch in range(0,img.shape[0]):
                    tformed_slice[ch] = ants.from_numpy(img[ch,:,:])
                combined_slice.append(ants.merge_channels(tformed_slice).numpy().astype('uint16'))
                continue
        else:
            img = img_volume.data[0][:,slice_idx,:,:]

        tformed_slice = None

        # Enlarge mask in histology space
        expanded_mov_group = expand_labels(mask_volume[slice_idx,:,:], distance=30)
        expanded_fix_group = expand_labels(fix_volume[slice_idx,:,:], distance=30)

        # For each group get mask and process mirror flip
        group_list = np.unique(fix_volume[slice_idx,:,:]).astype('uint8')
        group_list = np.delete(group_list,0)

        if sum(sum(fix_volume[slice_idx,:,:])) == 0:
            combined_slice.append(np.zeros((np.shape(expanded_fix_group)[-2],np.shape(expanded_fix_group)[-1],5), 'uint16'))
            continue

        for group_id in group_list:
            if group_id == 0:
                continue

            # Check whether the image needs to be flipped
            if group_flip != None :
                is_flip = group_id in group_flip[slice_idx]
            else:
                is_flip = False
            if group_error != None:
                is_error = group_id in group_error[slice_idx]
            else:
                is_error = False

            # Label refers to tight image mask
            # mask refers to enlarged image mask using expand label function
            mov_label = mask_volume[slice_idx,:,:]
            mov_mask  = expanded_mov_group
            fix_mask  = expanded_fix_group

            original_group_id = group_id
            if is_flip:
                mov_label = np.flipud(mov_label)
                mov_mask = np.flipud(mov_mask)
                original_group_id = get_flip_id(group_id, np.unique(mov_label).astype('uint8'))
            if is_error:
                if slice_idx > 167:
                    mov_label = np.fliplr(mov_label)
                    mov_mask = np.fliplr(mov_mask)
                else:
                    mov_label = np.flipud(mov_label)
                    mov_mask = np.flipud(mov_mask)


            # ants_mov_label = ants.from_numpy(mov_label)
            ants_mov_mask = ants.from_numpy(mov_mask)
            ants_fix_mask = ants.from_numpy(fix_mask)

            # Process masks
            ants_mov_mask = ants.get_mask(ants_mov_mask, low_thresh = original_group_id, high_thresh = original_group_id, cleanup=0)
            ants_mov_mask = ants.morphology(ants_mov_mask, operation='close', radius=10, mtype='binary', shape='ball')
            ants_mov_mask = ants.iMath(ants_mov_mask, 'FillHoles')
            ants_mov_mask = ants.get_mask(ants_mov_mask, low_thresh = 1, cleanup=0)

            ants_fix_mask = ants.get_mask(ants_fix_mask, low_thresh = group_id, high_thresh = group_id, cleanup=0)
            ants_fix_mask = ants.morphology(ants_fix_mask, operation='close', radius=10, mtype='binary', shape='ball')
            ants_fix_mask = ants.iMath(ants_fix_mask, 'FillHoles')
            ants_fix_mask = ants.get_mask(ants_fix_mask, low_thresh = 1, cleanup=0)

            tform_name = os.path.join(slice_folder, f'{slice_folder_idx}_{group_id}.mat')
            if not os.path.exists(tform_name):
                continue

            tformed_region = []
            for ch in range(0,img.shape[0]):
                channel_img = img[ch,:,:]
                if is_flip:
                    channel_img = np.flipud(channel_img)
                if is_error:
                    if slice_idx > 167:
                        channel_img = np.fliplr(channel_img)
                    else:
                        channel_img = np.flipud(channel_img)


                ants_channel_img = ants.from_numpy(channel_img.astype('uint32'))
                ants_channel_img = ants.mask_image(ants_channel_img, ants_mov_mask)

                ants_result_img = ants.apply_transforms(fixed=ants_fix_mask, moving=ants_channel_img,
                                                        transformlist=tform_name, interpolator='nearestNeighbor')
                ants_result_img = ants.mask_image(ants_result_img, ants_fix_mask)

                tformed_region.append(ants_result_img)

                if tformed_slice != None:
                    prev_img = tformed_slice[ch].numpy()
                    new_img = prev_img + ants_result_img.numpy()

                    tformed_slice[ch] = ants.from_numpy(new_img)

            if tformed_slice == None:
                tformed_slice = tformed_region

        for ch in range(0,img.shape[0]):
            channel_img = img[ch,:,:]
            if is_flip:
                channel_img = np.flipud(channel_img)
            if is_error:
                channel_img = np.flipud(channel_img)

            is_background = tformed_slice[ch].numpy() == 0
            is_background = ants.iMath_MD(ants.from_numpy(is_background.astype('uint32')), radius = 3).numpy()>0

            background_level = np.logical_and(np.logical_and(mov_mask>0, mov_label==0), (img[ch,:,:] > 0))
            background_level = np.median(channel_img[background_level])

            background_filled = tformed_slice[ch].numpy()
            # backgroun_filled[is_background] = background_level
            background_filled[is_background] = 0
            tformed_slice[ch] = ants.from_numpy(background_filled)

        if combined_slice == None:
            combined_slice.append(np.zeros(combined_slice[0].shape, 'uint16'))
        else:
            combined_slice.append(ants.merge_channels(tformed_slice).numpy().astype('uint16'))
        # print(combined_slice[slice_idx].shape)

    combined_volume = np.stack(combined_slice, axis=0)
    combined_volume = np.moveaxis(combined_volume, -1, 0)
    combined_volume = np.moveaxis(combined_volume, -1, -2)

    img_util.write_img(result_filename, combined_volume)


def transform_single_raw_image(slice_idx: int, czi_folder: str, tform_folder: str, lemur_prefix: str,
                               src_height: int, src_width: int, des_height: int,
                               des_width: int, group_flip:list = None, brain_info: dict = None):
    slice_folder_idx = slice_idx + 1
    slice_folder = os.path.join(tform_folder, str(slice_folder_idx))

    if brain_info is None:
        scene_idx = (slice_idx) % 4 + 1
        file_idx = math.ceil(slice_folder_idx / 4)
    else:
        scene_idx = int(brain_info['scene'][slice_idx])
        file_idx = int(brain_info['filename'][slice_idx][-6:-4])
    if scene_idx < 0:
        return

    # Get CZI histology image
    nim_name = os.path.join(czi_folder, 'background_corrected', f'{lemur_prefix}_{file_idx:02}_scene'
                                                                f'{scene_idx}_background_corrected.nim')
    if brain_info is None:
        pass
    else:
        filename = os.path.basename(brain_info['filename'][slice_idx])[:-4]
        nim_name = os.path.join(czi_folder, 'background_corrected', f'{lemur_prefix}_{file_idx:02}_scene{scene_idx}_background_corrected.nim')


    result_filename = os.path.join(czi_folder, 'background_corrected', 'aligned',
                                   f'{lemur_prefix}_{file_idx:02}_scene'
                                   f'{scene_idx}_aligned.nim')
    if (os.path.exists(result_filename)):
        logger.info(f'{result_filename} Result exists')
        return

    if os.path.exists(nim_name):
        imgInfo = ZImg(nim_name, region=ZImgRegion())
        img = imgInfo.data[0][:, 0, :, :]
    else:
        logger.info(f'{nim_name} does not exist')
        return

    logger.info(f'Running {lemur_prefix}_{file_idx:02}_scene{scene_idx}')

    if group_flip != None:
        is_flip = 1 in group_flip[slice_idx]
    else:
        is_flip = False
    tform_name = os.path.join(slice_folder, f'{slice_folder_idx}_1.mat')
    if not os.path.exists(tform_name):
        return
    tform_data = scipy.io.loadmat(tform_name)
    tform_param = tform_data['AffineTransform_float_2_2']
    tform_affine = np.float32([[tform_param[0], tform_param[1], tform_param[4]],
                               [tform_param[2], tform_param[3], tform_param[5]],
                               [0, 0, 1]])
    tform_cent = np.float32([[1, 0, tform_data['fixed'][0]], [0, 1, tform_data['fixed'][1]], [0, 0, 1]])
    tform = np.dot(np.dot(tform_cent, tform_affine), np.linalg.inv(tform_cent))
    tform[0:2, 2] = tform[0:2, 2] * 16

    tformed_slice = np.zeros(shape=(img.shape[0], 1, des_height, des_width), dtype='uint16')
    for ch in range(0, img.shape[0]):
        channel_img = img[ch, :, :]
        if is_flip:
            channel_img = np.pad(channel_img,((0, max(0,src_height-img.shape[1])), (0,max(0,src_width-img.shape[2]))))
            channel_img = channel_img[:src_height, :src_width]
            channel_img = np.flipud(channel_img)
        channel_img = np.moveaxis(channel_img, -1, -2)
        tformed_slice[ch, 0, :, :] = cv2.warpAffine(channel_img, np.linalg.inv(tform)[0:2, :], dsize=(des_width,
                                                             des_height),
                                                    flags=cv2.INTER_LINEAR)
    pathlib.Path(os.path.join(czi_folder, 'background_corrected', 'aligned')).mkdir(parents=True, exist_ok=True)
    des_height

def apply_rigid_transform_image_final(result_filename: str, tform_folder: str, reflabel_name: str, movlabel_name: str,
                                    czi_folder: str, lemur_prefix: str, *,
                                    group_flip: list = None, czi_filename: str = None,
                                    brain_info: dict = None):
    # mask_img = ZImg(movlabel_name)
    # mask_volume = mask_img.data[0][0, :, :, :]
    #
    fix_img = ZImg(reflabel_name)
    fix_volume = fix_img.data[0][0,:,:,:]
    fix_volume = np.moveaxis(fix_volume,-2,-1)
    des_height = fix_volume.shape[1] * 16
    des_width = fix_volume.shape[2] * 16
    #
    # infoList = ZImg.readImgInfos(movlabel_name)
    # assert len(infoList) == 1 and infoList[0].numTimes == 1
    # img_info = infoList[0]
    #
    # img_rotate = True if img_info.height > img_info.width else False
    # if not img_rotate:
    #     mask_volume = np.moveaxis(mask_volume, -2, -1)
    #
    # combined_slice = []

    # def transform_single_raw_image(slice_idx: int, tform_folder:str, lemur_prefix:str, des_height: int,  des_width:int,
    # brain_info: dict=None):
    #     logger.info(f'running {slice_idx}')
    #     slice_folder_idx = slice_idx + 1
    #     slice_folder = os.path.join(tform_folder, str(slice_folder_idx))
    #
    #     if brain_info is None:
    #         scene_idx = (slice_idx) % 4 + 1
    #         file_idx = math.ceil(slice_folder_idx / 4)
    #     else:
    #         scene_idx = int(brain_info['scene'][slice_idx])
    #         file_idx = slice_idx
    #     if scene_idx < 0:
    #         return
    #
    #     # Get CZI histology image
    #     if brain_info is None:
    #         nim_name = os.path.join(czi_folder, 'background_corrected', f'{lemur_prefix}_{file_idx:02}_scene'
    #                                             f'{scene_idx}_background_corrected.nim')
    #     else:
    #         filename = os.path.basename(brain_info['filename'][file_idx])[:-4]
    #         nim_name = os.path.join(czi_folder, f'{filename}_scene{scene_idx}_background_corrected.nim')
    #     if os.path.exists(nim_name):
    #         imgInfo = ZImg(nim_name, region=ZImgRegion())
    #         img = imgInfo.data[0][:, 0, :, :]
    #     else:
    #         return
    #
    #     if group_flip != None:
    #         is_flip = 1 in group_flip[slice_idx]
    #     else:
    #         is_flip = False
    #     tform_name = os.path.join(slice_folder, f'{slice_folder_idx}_1.mat')
    #     if not os.path.exists(tform_name):
    #         return
    #     tform_data = scipy.io.loadmat(tform_name)
    #     tform_param = tform_data['AffineTransform_float_2_2']
    #     tform_affine = np.float32([[tform_param[0], tform_param[1], tform_param[4]],
    #                               [tform_param[2], tform_param[3], tform_param[5]],
    #                               [0, 0, 1]])
    #     tform_cent = np.float32([[1, 0, tform_data['fixed'][0]], [0, 1, tform_data['fixed'][1]], [0, 0, 1]])
    #     tform = np.dot(np.dot(tform_cent, tform_affine), np.linalg.inv(tform_cent))
    #     tform[0:2, 2] = tform[0:2, 2] * 16
    #
    #     tformed_slice = np.zeros(shape=(img.shape[0], des_width, des_height), dtype='uint16')
    #     for ch in range(0, img.shape[0]):
    #         channel_img = img[ch, :, :]
    #         if is_flip:
    #             channel_img = np.flipud(channel_img)
    #         tformed_slice[ch,:,:] = cv2.warpAffine(channel_img, np.linalg.inv(tform)[0:2, :], dsize=(des_width,
    #                                                                                               des_height),
    #                                                flags=cv2.INTER_LINEAR)
    #     infoList = ZImg.readImgInfos(nim_name)
    #     infoList = infoList[0]
    #     infoList.width = tformed_slice.shape[-1]
    #     infoList.height = tformed_slice.shape[-2]
    #     img = ZImg(tformed_slice, infoList)
    #     result_filename = os.path.join(czi_folder, 'background_corrected', 'aligned', f'{lemur_prefix}_{file_idx:02}_scene'
    #                                             f'{scene_idx}_aligned.nim')
    #     img.save(result_filename)
    #     logger.info(f'image {slice_idx} done')
    #
    #
    # def pool_transform_single_raw_image(img_idx: int):
    #     logger.info(f'running {img_idx}')
    #     # transform_single_raw_image(img_idx, tform_folder, lemur_prefix, des_height, des_width, brain_info)
    #     test(img_idx)
    #
    #
    # with multiprocessing.Pool(4) as pool:
    #     logger.info(f'running')
    #     pool.map_async(pool_transform_single_raw_image, range(10),chunksize=1, callback=None).wait()
    #     logger.info(f'done')

    # for slice_idx in range(mask_volume.shape[0]):
    # # for slice_idx in range(5):
    #     logger.info(f'running {slice_idx}')
    #     slice_folder_idx = slice_idx + 1
    #     slice_folder = os.path.join(tform_folder, str(slice_folder_idx))
    #
    #     if brain_info is None:
    #         scene_idx = (slice_idx) % 4 + 1
    #         file_idx = math.ceil(slice_folder_idx / 4)
    #     else:
    #         scene_idx = int(brain_info['scene'][slice_idx])
    #         file_idx = slice_idx
    #     if scene_idx < 0:
    #         img = np.zeros((5, mask_volume.shape[1], mask_volume.shape[2]), 'float32')
    #         for ch in range(0, img.shape[0]):
    #             tformed_slice[ch] = ants.from_numpy(img[ch, :, :])
    #         combined_slice.append(ants.merge_channels(tformed_slice).numpy().astype('uint16'))
    #         continue
    #
    #     # Get CZI histology image
    #     if brain_info is None:
    #         nim_name = os.path.join(czi_folder, 'background_corrected', f'{lemur_prefix}_{file_idx:02}_scene'
    #                                             f'{scene_idx}_background_corrected.nim')
    #     else:
    #         filename = os.path.basename(brain_info['filename'][file_idx])[:-4]
    #         nim_name = os.path.join(czi_folder, f'{filename}_scene{scene_idx}_background_corrected.nim')
    #
    #     if os.path.exists(nim_name):
    #         imgInfo = ZImg(nim_name, region=ZImgRegion())
    #         img = imgInfo.data[0][:, 0, :, :]
    #         if not img_rotate:
    #             img = np.moveaxis(img, -2, -1)
    #         x_pad = mask_volume.shape[1] * 16 - img.shape[1]
    #         y_pad = mask_volume.shape[2] * 16 - img.shape[2]
    #
    #         if x_pad > 0:
    #             img = np.pad(img, ((0, 0), (0, x_pad), (0, 0)))
    #         else:
    #             img = img[:, :x_pad, :]
    #
    #         if y_pad > 0:
    #             img = np.pad(img, ((0, 0), (0, 0), (0, y_pad)))
    #         else:
    #             img = img[:, :, :y_pad]
    #     else:
    #         tformed_slice = [[]] * 5
    #         img = np.zeros((5, fix_volume.shape[1]*16, fix_volume.shape[2]*16), 'float32')
    #         for ch in range(0, img.shape[0]):
    #             tformed_slice[ch] = ants.from_numpy(img[ch, :, :])
    #         combined_slice.append(ants.merge_channels(tformed_slice).numpy().astype('uint16'))
    #         continue
    #
    #     tformed_slice = None
    #
    #     # Enlarge mask in histology space
    #     expanded_mov_group = expand_labels(mask_volume[slice_idx, :, :], distance=30)
    #
    #     # For each group get mask and process mirror flip
    #     group_list = np.unique(fix_volume[slice_idx, :, :]).astype('uint8')
    #     group_list = np.delete(group_list, 0)
    #
    #     for group_id in group_list:
    #         if group_id == 0:
    #             continue
    #         # Check whether the image needs to be flipped
    #         if group_flip != None:
    #             is_flip = group_id in group_flip[slice_idx]
    #         else:
    #             is_flip = False
    #         if group_error != None:
    #             is_error = group_id in group_error[slice_idx]
    #         else:
    #             is_error = False
    #
    #         # Label refers to tight image mask
    #         # mask refers to enlarged image mask using expand label function
    #         fix_label = fix_volume[slice_idx, :, :]
    #         mov_mask = expanded_mov_group
    #
    #         original_group_id = group_id
    #         if is_flip:
    #             mov_mask = np.flipud(mov_mask)
    #             original_group_id = get_flip_id(group_id, np.unique(mov_mask).astype('uint8'))
    #         if is_error:
    #             if slice_idx > 167:
    #                 mov_mask = np.fliplr(mov_mask)
    #             else:
    #                 mov_mask = np.flipud(mov_mask)
    #
    #         # ants_mov_label = ants.from_numpy(mov_label)
    #         ants_mov_mask = ants.from_numpy(mov_mask)
    #         ants_fix_label = ants.from_numpy(fix_label.astype('float32'))
    #
    #         # ants_mov_mask.set_spacing((16, 16))
    #         # ants_fix_label.set_spacing((16, 16))
    #
    #         # Process masks
    #         ants_mov_mask = ants.get_mask(ants_mov_mask, low_thresh=original_group_id, high_thresh=original_group_id,
    #                                       cleanup=0)
    #         ants_mov_mask = ants.morphology(ants_mov_mask, operation='close', radius=10, mtype='binary', shape='ball')
    #         ants_mov_mask = ants.iMath(ants_mov_mask, 'FillHoles')
    #         ants_mov_mask = ants.get_mask(ants_mov_mask, low_thresh=1, cleanup=0)
    #
    #         # rescale to original size
    #         ants_mov_mask = ants_mov_mask.resample_image((1/16, 1/16))
    #         ants_fix_label = ants_fix_label.resample_image((1/16, 1/16))
    #
    #         tform_name = os.path.join(slice_folder, f'{slice_folder_idx}_{group_id}.mat')
    #         tform = ants.read_transform(tform_name)
    #         # tform.set_parameters(np.append(tform.parameters[:-2], [tform.parameters[-2] * 16, tform.parameters[-1] * 16]))
    #         # tform_name = os.path.join(slice_folder, f'{slice_folder_idx}_{group_id}_original.mat')
    #         # ants.write_transform(tform, tform_name)
    #
    #         if not os.path.exists(tform_name):
    #             continue
    #
    #         tformed_slice = []
    #         for ch in range(0, img.shape[0]):
    #             channel_img = img[ch, :, :]
    #             if is_flip:
    #                 channel_img = np.flipud(channel_img)
    #             if is_error:
    #                 if slice_idx > 167:
    #                     channel_img = np.fliplr(channel_img)
    #                 else:
    #                     channel_img = np.flipud(channel_img)
    #
    #             ants_channel_img = ants.from_numpy(channel_img.astype('uint32'))
    #             ants_channel_img.set_spacing((1/16, 1/16))
    #             ants_channel_img = ants.mask_image(ants_channel_img, ants_mov_mask)
    #
    #             ants_result_img = ants.apply_transforms(fixed=ants_fix_label, moving=ants_channel_img,
    #                                                     transformlist=tform_name, interpolator='nearestNeighbor')
    #             tformed_slice.append(ants_result_img.numpy())
    #
    #     if combined_slice == None:
    #         combined_slice.append(np.zeros(combined_slice[0].shape, 'uint16'))
    #     else:
    #         combined_slice.append(np.stack(tformed_slice, 0).astype('uint16'))
    #     # print(combined_slice[slice_idx].shape)
    #
    # combined_volume = np.stack(combined_slice, axis=1)
    # combined_volume = np.moveaxis(combined_volume, -1, -2)
    #
    # infoList = ZImg.readImgInfos(nim_name)
    # infoList = infoList[0]
    # infoList.depth = combined_volume.shape[1]
    # infoList.width = combined_volume.shape[-1]
    # infoList.height = combined_volume.shape[-2]
    # img = ZImg(combined_volume, infoList)
    # img.save(result_filename)

    # img_util.write_img(result_filename, combined_volume)±


def run_SyN_transform_slice(mov_folder: str, reflabel_name: str, group_split: list):
    fix_volume = ants.image_read(reflabel_name)

    # Read annotation and get dict (nx2 array)
    (_, folder_list, _) = next(os.walk(os.path.join(mov_folder)))

    # Iterate over each region > slice > shape
    for folder_name in folder_list:
        slice_idx = int(folder_name)

        (_, _, filenames) = next(os.walk(os.path.join(mov_folder, f"{slice_idx}")))
        for matching in [s for s in filenames if "mhd" in s]:
            mov_filename = os.path.join(mov_folder, f"{slice_idx}", matching)
            fwd_filename = os.path.join(mov_folder, f"{slice_idx}", f"{matching[0:-4]}_fwd.h5")
            # inv_filename = os.path.join(mov_folder, f"{slice_idx}", f"{matching[0:-4]}_inv.h5")
            # if os.path.exists(inv_filename):
            #     continue

            mov = ants.image_read(mov_filename)
            mov = ants.get_mask(mov, low_thresh=1, high_thresh=None, cleanup=0)
            mov = ants.morphology(mov, operation='close', radius=10, mtype='binary', shape='ball')
            mov = ants.iMath(mov, 'FillHoles')

            group_id = int(re.split('^([0-9]+)_([0-9]+)', matching)[2])
            need_split = group_id in group_split[slice_idx-1]

            fix = fix_volume.slice_image(2, slice_idx-1)
            fix = ants.get_mask(fix, low_thresh = group_id, high_thresh = group_id, cleanup=0)

            if not need_split:
                fix = ants.morphology(fix, operation='close', radius=20, mtype='binary', shape='ball')
                fix = ants.iMath(fix, 'FillHoles')

            if slice_idx in [161,162] and group_id in [1,11]:
                fix = fix_volume.slice_image(2, slice_idx-2)
                fix = ants.get_mask(fix, low_thresh = group_id, high_thresh = group_id, cleanup=0)

            fix = ants.morphology(fix, operation='close', radius=5, mtype='binary', shape='ball')
            fix = ants.iMath(fix, 'FillHoles')

            fix = ants.resample_image(fix, (10,10), 0, 0)
            mov = ants.resample_image(mov, (10,10), 0, 0)

            logger.info(f'Currently running {slice_idx}_{group_id}')

            mytx = ants.registration(fixed=fix , moving=mov, type_of_transform='antsRegistrationSyNQuick[bo]',
                                     grad_step=0.5, write_composite_transform=True, verbose=False, syn_transform = "BSplineSyN[0.05,500,0,2]")
            inv_filename = os.path.join(mov_folder, f"{slice_idx}", f"{matching[0:-4]}_inv_global.h5")
            fwd_filename = os.path.join(mov_folder, f"{slice_idx}", f"{matching[0:-4]}_fwd_global.h5")
            shutil.copyfile(mytx['invtransforms'], inv_filename)
            shutil.copyfile(mytx['fwdtransforms'], inv_filename)

            mytx = ants.registration(fixed=fix , moving=mytx['warpedmovout'], type_of_transform='antsRegistrationSyNQuick[bo]',
                                     grad_step=0.5, write_composite_transform=True, verbose=False, syn_transform = "BSplineSyN[0.05,25,0,2]")
            inv_filename = os.path.join(mov_folder, f"{slice_idx}", f"{matching[0:-4]}_inv_local.h5")
            fwd_filename = os.path.join(mov_folder, f"{slice_idx}", f"{matching[0:-4]}_fwd_local.h5")
            shutil.copyfile(mytx['invtransforms'], inv_filename)
            shutil.copyfile(mytx['fwdtransforms'], inv_filename)


def apply_SyN_transform_image(result_filename: str, tform_folder: str, fix_folder: str):
    # Get all subfolder and the length
    slice_list = glob.glob(os.path.join(tform_folder, "*"))

    # Iterate over each slice > region > channel
    for slice_folder in slice_list:
        logger.info(f'running {slice_folder}')
        pathlib.Path(os.path.join(slice_folder, "deform")).mkdir(parents=True, exist_ok=True)
        tform_list = [os.path.basename(x) for x in glob.glob(os.path.join(slice_folder, "*_fwd.h5"))]


        for tform in tform_list:
            img_slice = tform.split("_")[0]
            shape_id = tform.split("_")[1]

            deform_filename = os.path.join(slice_folder, f"{tform}")

            fix_filename = os.path.join(fix_folder, f"{img_slice}", f"{img_slice}_{shape_id}.mhd")
            fix_img = ants.image_read(fix_filename)
            fix_img = ants.get_mask(fix_img, low_thresh=1, high_thresh=None, cleanup=0)
            fix_img = ants.morphology(fix_img, operation='close', radius=10, mtype='binary', shape='ball')
            fix_img = ants.iMath(fix_img, 'FillHoles')

            # apply transform to signal
            signal_list = [os.path.basename(x) for x in glob.glob(os.path.join(slice_folder,f"signal_{img_slice}_{shape_id}_*.mhd"))]
            signal_list.sort()

            deform_signal = []
            for signal_name in signal_list:
                signal_ch = signal_name.split("_")[3].split(".")[0]
                mov_filename = os.path.join(slice_folder,f"{signal_name}")
                mov_img = ants.image_read(mov_filename)

                deform_signal.append(ants.apply_transforms(fixed=fix_img, moving=mov_img, transformlist=deform_filename))
            ants.image_write(ants.merge_channels(deform_signal), os.path.join(slice_folder, "deform", f"deform_{img_slice}_{shape_id}.nii"))


def run_SyN_transform_slice_from_volume(tform_folder: str, fixlabel_name: str, movlabel_name: str):
    img = ZImg(movlabel_name)
    mov_volume = img.data[0][0,:,:,:]
    fix_volume = ants.image_read(fixlabel_name)

    infoList = ZImg.readImgInfos(movlabel_name)
    assert len(infoList) == 1 and infoList[0].numTimes == 1
    img_info = infoList[0]

    img_rotate = True if img_info.height > img_info.width else False
    if not img_rotate:
        mov_volume = np.moveaxis(mov_volume,-2,-1)

    for slice_idx in range(fix_volume.shape[2]):
    # for slice_idx in range(19,25):
        slice_folder_idx = slice_idx + 1
        deform_folder = os.path.join(tform_folder, str(slice_folder_idx))
        pathlib.Path(deform_folder).mkdir(parents=True, exist_ok=True)

        for group_id in np.unique(fix_volume.slice_image(2, slice_idx).numpy()).astype('uint8'):
            if group_id == 0:
                continue

            mov_slice = mov_volume[slice_idx,:,:]
            mov_slice = ants.from_numpy(mov_slice)

            mov = ants.get_mask(mov_slice, low_thresh = group_id, high_thresh = group_id, cleanup=0)
            mov = ants.morphology(mov, operation='close', radius=10, mtype='binary', shape='ball')
            mov = ants.iMath(mov, 'FillHoles')

            if np.sum(mov.numpy()) == 0:
                continue

            fix = fix_volume.slice_image(2, slice_idx)
            fix = ants.get_mask(fix, low_thresh = group_id, high_thresh = group_id, cleanup=0)
            fix = fix.numpy()
            fix[1:math.ceil(fix.shape[0]/2), :] = 0.0
            fix = ants.from_numpy(fix)

            fix = ants.resample_image(fix, (5,5), 0, 0)
            mov = ants.resample_image(mov, (5,5), 0, 0)

            logger.info(f'Currently running {slice_idx}_{group_id}')
            mytx = ants.registration(fixed=fix , moving=mov, type_of_transform='antsRegistrationSyNQuick[bo]',
                                     grad_step=0.5, write_composite_transform=True, verbose=False, syn_transform = "BSplineSyN[0.05,500,0,2]")
            inv_filename = os.path.join(deform_folder,  f"{slice_idx}_{group_id}_inv_global.h5")
            fwd_filename = os.path.join(deform_folder,  f"{slice_idx}_{group_id}_fwd_global.h5")
            shutil.copyfile(mytx['invtransforms'], inv_filename)
            shutil.copyfile(mytx['fwdtransforms'], fwd_filename)

            mytx = ants.registration(fixed=fix , moving=mytx['warpedmovout'], type_of_transform='antsRegistrationSyN[bo]',
                                     grad_step=0.5, write_composite_transform=True, verbose=False, syn_transform = "BSplineSyN[0.05,25,0,2]")
            inv_filename = os.path.join(deform_folder, f"{slice_folder_idx}_{group_id}_inv_local.h5")
            fwd_filename = os.path.join(deform_folder, f"{slice_folder_idx}_{group_id}_fwd_local.h5")
            shutil.copyfile(mytx['invtransforms'], inv_filename)
            shutil.copyfile(mytx['fwdtransforms'], fwd_filename)


def apply_SyN_transform_annotation(ra_filename: str, region_group: list, group_split: list,
                                   result_filename: str, tform_folder: str,
                                     *, midline_filename: str = None, midline_x: float = None, downsample_ratio: int = 16):
    # Read annotation and get dict (nx2 array)
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    logger.info(f'finish reading {ra_filename}')
    # Read midline and get dict (nx2 array)
    if midline_filename != None:
        mid_dict = region_annotation.read_region_annotation(midline_filename)
        logger.info(f'finish reading {midline_filename}')

    # Iterate over each region > slice > shape
    mid_slice = {}
    for region_id_, region in ra_dict['Regions'].items():
        if region['ROI'] is not None:
            if region_id_ == -1:
                region_id = 315
            else:
                region_id = region_id_

            if region_id == 4775:
                region['ParentID'] = 1097

            for slice_idx, sliceROIs in region['ROI']['SliceROIs'].items():
                slice_folder_idx = slice_idx + 1
                if slice_idx not in mid_slice.keys():
                    mid_slice[slice_idx] = get_slice_centroid(ra_dict, slice_idx, 0)

                mapped_region_id = 0
                # Get the group ID for the region
                for group_idx, group in enumerate(region_group[slice_idx], start=1):
                    if region_id in group:
                        mapped_region_id = group_idx
                        break
                    if region['ParentID'] in group:
                        mapped_region_id = group_idx
                        break

                # If the region is not mapped, ignore
                if mapped_region_id == 0:
                    continue

                # Check the region in the slice need cutting
                need_split = mapped_region_id in group_split[slice_idx]

                logger.info(f'Currently running {slice_idx}_{group_idx}')

                for shape in sliceROIs:
                    for subShape in shape:
                        shape_id = mapped_region_id;

                        if(need_split):
                            # Get average midline and check whehter the shape is on left/right hemisphere
                            mid_x = np.mean(subShape['Points'][:,0])
                            if midline_filename != None:
                                if(mid_x > get_midline_average(mid_dict, slice_idx)):
                                    shape_id = mapped_region_id + 10
                            elif midline_x != None:
                                if(mid_x > midline_x):
                                    shape_id = mapped_region_id + 10
                            else:
                                if(mid_x > mid_slice[slice_idx]):
                                    shape_id = mapped_region_id + 10

                        # Get annotation file name
                        deform_filename = os.path.join(tform_folder, f"{slice_folder_idx}", f"{slice_folder_idx}_{shape_id}_inv_local.h5")
                        if os.path.exists(deform_filename):
                            # Read affine elastix transform
                            def transform_fun(input_coords: np.ndarray, deform_filename:str):
                                assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape

                                res = pd.DataFrame(data=input_coords.copy(), columns=['x', 'y'])
                                res = res/downsample_ratio
                                res = ants.apply_transforms_to_points(2, res, deform_filename)
                                res = res * downsample_ratio
                                res = res.to_numpy()

                                assert res.ndim == 2 and res.shape[1] == 2, res.shape
                                return res

                            # Apply transform
                            deform_filename = os.path.join(tform_folder, f"{slice_folder_idx}", f"{slice_folder_idx}_{shape_id}_inv_local.h5")
                            subShape['Points'] = transform_fun(subShape['Points'], deform_filename)

                            deform_filename = os.path.join(tform_folder, f"{slice_folder_idx}", f"{slice_folder_idx}_{shape_id}_inv_global.h5")
                            subShape['Points'] = transform_fun(subShape['Points'], deform_filename)

                        else:
                            subShape = None

    # Write Region Annotation
    region_annotation.write_region_annotation_dict(ra_dict, result_filename)


def pool_transform_single_raw_image(parameter_tuple: list):
    transform_single_raw_image(parameter_tuple[0], czi_folder=parameter_tuple[1],
                               tform_folder=parameter_tuple[2],
                               lemur_prefix=parameter_tuple[3], group_flip=parameter_tuple[4],
                               src_height=parameter_tuple[5], src_width=parameter_tuple[6], brain_info=parameter_tuple[7],
                               des_height=20288, des_width=28080)


if __name__ == "__main__":

    folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                          '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
    lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')

    folder = os.path.join(lemur_folder, 'Hotsauce_334A', '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
    img_filename = os.path.join(folder, 'hj_aligned', 'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    midline_filename = os.path.join(folder, 'interns_edited_results', 'sh_cut_in_half.reganno')

    hyungju_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align'

    result_list = ['Fig_PV_TH_NeuN',
                   'Fig_SMI99_NeuN_VGlut2',
                   'Garlic_SMI99_VGluT2_M2',
                   'Hotsauce_PV_TH_NeuN',
                   'Icecream_PV_TH_NeuN',
                   'Icecream_SMI99_NeuN_VGlut2',
                   'Jellybean_FOXP2_SMI32_NeuN',
                   'Jellybean_vGluT2_SMI32_vGluT1']
    folder_list = ['Fig_325AA/180918_Lemur-Fig_PV_TH_NeuN',
                   'Fig_325AA/180914_fig_SMI99_NeuN_VGlut2',
                   'Garlic_320CA/181023_Lemur-Garlic_SMI99_VGluT2_M2',
                   'Hotsauce_334A/181016_Lemur-Hotsauce_PV_TH_NeuN',
                   'Icecream_225BD/190221_icecream_PV_TH_NeuN',
                   'Icecream_225BD/20190218_icecream_SMI99_NeuN_VGlut2',
                   'Jellybean_289BD/20190813_jellybean_FOXP2_SMI32_NeuN',
                   'Jellybean_289BD/20190827_jellybean_vGluT2_SMI32_vGluT1']
    prefix_list = ['Lemur-F_PV_TH_NeuN',
                   'Lemur-F_SMI99_NeuN_VGlut2',
                   'Lemur-G_SMI99_VGluT2_M2',
                   'Lemur-H_PV_TH_NeuN',
                   'Lemur-I_PV_TH_NeuN',
                   'Lemur-I_SMI99_VGluT2_NeuN',
                   'Lemur-J_FOXP2_SMI32_NeuN',
                   'Lemur-J_vGluT2_SMI32_vGluT1']

    run_brain_info = False
    # for idx in range(8):
    for idx in [3]:
        result_folder = os.path.join(hyungju_folder, 'alignment-no-split', result_list[idx])
        czi_folder = os.path.join(lemur_folder, folder_list[idx])

        # ---------------------------------------------------------------------------------------------------------------------------
        # 00. Generate group and split info
        # ---------------------------------------------------------------------------------------------------------------------------
        # if not os.path.exists(os.path.join(result_folder, 'blockface_group.txt')):
        #     blockface_group_name = os.path.join(result_folder, 'blockface_group.txt')
        #     blockface_split_name = os.path.join(result_folder, 'blockface_split.txt')
        #     blockface_filename = os.path.join(result_folder, '00_blockface_label.tif')
        #     get_blockface_status(blockface_filename, blockface_group_name, blockface_split_name)

        if run_brain_info:
            stacked_label_filename = os.path.join(result_folder, '00_stacked_label.nim')
            stacked_ra_filename = os.path.join(result_folder, '00_stacked_annotation.reganno')
            region_group_name = os.path.join(result_folder, 'region_group.txt')

            get_group_status(stacked_label_filename, region_group_name)

            continue

        # ---------------------------------------------------------------------------------------------------------------------------
        # 00. Read group and split info
        # ---------------------------------------------------------------------------------------------------------------------------

        # reference_group_name = os.path.join(result_folder, 'reference_group.txt')
        # with open(reference_group_name) as json_file:
        #     reference_group = json.load(json_file)
        #
        # reference_split_name = os.path.join(result_folder, 'reference_split.txt')
        # with open(reference_split_name) as json_file:
        #     reference_split = json.load(json_file)

        # blockface_group_name = os.path.join(result_folder, 'blockface_group.txt')
        # with open(blockface_group_name) as json_file:
        #     blockface_group = json.load(json_file)

        # blockface_split_name = os.path.join(result_folder, 'blockface_split.txt')
        # with open(blockface_split_name) as json_file:
        #     blockface_split = json.load(json_file)

        # region_group_name = os.path.join(result_folder, 'region_group.txt')
        # with open(region_group_name) as json_file:
        #     region_group = json.load(json_file)
        #
        # group_split_name = os.path.join(result_folder, 'group_split.txt')
        # with open(group_split_name) as json_file:
        #     group_split = json.load(json_file)
        #
        group_flip_name = os.path.join(result_folder, 'group_flip.txt')
        with open(group_flip_name) as json_file:
            group_flip = json.load(json_file)
        #
        # group_error_name = os.path.join(result_folder, 'group_error.txt')
        # with open(group_error_name) as json_file:
        #     group_error = json.load(json_file)
        #
        brain_info_name = os.path.join(czi_folder, 'info.txt')
        brain_info = None
        if os.path.exists(brain_info_name):
            brain_info = read_brain_info(brain_info_name)

        # ---------------------------------------------------------------------------------------------------------------------------
        # 00. Generate matching reference volume
        # ---------------------------------------------------------------------------------------------------------------------------
        reflabel_filename = os.path.join(hyungju_folder, 'all-dataset', 'Hotsauce_blockface_outline_grouped_fix_interpolated_mirror.tiff')
        # reflabel_filename = os.path.join(result_folder, '00_matched_reference.tif')
        stacked_label_filename = os.path.join(result_folder, '00_stacked_label.nim')
        result_filename = os.path.join(result_folder, '00_matched_reference.nim')

        # export_matching_reference(reflabel_filename, stacked_label_filename, result_filename, result_folder,
        #                           reference_group, region_group, reference_split)

        # ---------------------------------------------------------------------------------------------------------------------------
        # 01. Generate big-region group label
        # ---------------------------------------------------------------------------------------------------------------------------
        stacked_label_filename = os.path.join(result_folder, '00_stacked_label.nim')
        stacked_ra_filename = os.path.join(result_folder, '00_stacked_annotation.reganno')
        result_filename = os.path.join(result_folder, '01_grouped_label.nim')

        # export_grouped_label_img(stacked_label_filename, stacked_ra_filename, region_group, group_split, result_filename, group_flip = group_flip)

        # ---------------------------------------------------------------------------------------------------------------------------
        # 02. Compute linear transformation for alignment to blockface outline
        # ---------------------------------------------------------------------------------------------------------------------------
        reflabel_filename = os.path.join(result_folder, '00_matched_reference.nim')
        movlabel_filename = os.path.join(result_folder, '01_grouped_label.nim')
        result_filename = os.path.join(result_folder, '02_aligned_label.nim')
        tform_folder = os.path.join(result_folder, 'mov')

        # run_rigid_transform_slice(reflabel_filename, movlabel_filename, result_filename, tform_folder, is_scaling = False, group_flip = group_flip, group_error = group_error)

        # ---------------------------------------------------------------------------------------------------------------------------
        # 02. Apply linear transform to annotation and image
        # ---------------------------------------------------------------------------------------------------------------------------
        tform_folder = os.path.join(result_folder, 'mov')
        ra_filename = os.path.join(result_folder, '00_stacked_annotation.reganno')
        result_filename = os.path.join(result_folder, '02_aligned_annotation.reganno')

        # apply_rigid_transform_annotation(ra_filename, region_group, group_split, result_filename, tform_folder, midline_filename = midline_filename)

        reflabel_name = os.path.join(result_folder, '02_aligned_label.nim')
        movlabel_name = os.path.join(result_folder, '01_grouped_label.nim')
        result_filename = os.path.join(result_folder, '02_aligned_signal.tiff')
        lemur_prefix = prefix_list[idx]
        combined_info = ZImg.readImgInfos(os.path.join(czi_folder, '01_grouped_label.nim'))

        # apply_rigid_transform_image(result_filename, tform_folder, reflabel_name, movlabel_name, czi_folder,
        # lemur_prefix, group_split=group_split, run_shading_correction = False, group_flip = group_flip, group_error = group_error, brain_info = brain_info)


        result_filename = os.path.join(czi_folder, 'background_corrected', f'{lemur_prefix}_combined_corrected.nim')
        # apply_rigid_transform_image_final(result_filename, tform_folder, reflabel_name,
        #                                   movlabel_name, czi_folder, lemur_prefix,
        #                                   group_flip=group_flip, brain_info=brain_info)

        param_set = [(idx, czi_folder, tform_folder, lemur_prefix, group_flip, combined_info[0].height*16,combined_info[0].width*16,
                      brain_info) for idx in
                     range(200)]
        # pool_transform_single_raw_image(param_set[53])

        with multiprocessing.Pool(4) as pool:
            pool.map_async(pool_transform_single_raw_image, param_set, chunksize=1, callback=None).wait()


        # ---------------------------------------------------------------------------------------------------------------------------
        # 00-1. Generate matching blockface label
        # ---------------------------------------------------------------------------------------------------------------------------
        blockface_filename = os.path.join(result_folder, '00_blockface_label.tif')
        stacked_label_filename = os.path.join(result_folder, '00_stacked_label.nim')
        result_filename = os.path.join(result_folder, '00_matched_blockface.nim')


        # export_matching_blockface(reflabel_filename, stacked_label_filename, result_filename, result_folder, reference_group,  reference_split)