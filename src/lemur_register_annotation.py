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
from utils import shading_correction
from skimage.segmentation import watershed, expand_labels


logger = setup_logger()


def _callback(result):
    logger.info(f'finished {result}')

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
    
def run_rigid_transform_slice(reflabel_name: str, movlabel_name: str, result_filename: str, tform_folder: str, is_scaling: bool, group_flip:list = None, group_error:list = None):
    img = ZImg(movlabel_name)
    mov_volume = img.data[0][0,:,:,:].copy()
    
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
    # for slice_idx in range(7,8):
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
            if group_flip != None and group_error != None:
                is_flip = group_id in group_flip[slice_idx]
                is_error = group_id in group_error[slice_idx]
            else:
                is_flip = False
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
                    mov_slice = (np.flipud(mov_slice))
                    
                mov_slice = ants.from_numpy(mov_slice)
                    
                mov = ants.get_mask(mov_slice, low_thresh = group_id, high_thresh = group_id, cleanup=0)
                mov = ants.morphology(mov, operation='close', radius=10, mtype='binary', shape='ball')
                mov = ants.iMath(mov, 'FillHoles')
                
                if np.sum(mov.numpy()) < 10:
                    continue
                
                fix = fix_volume.slice_image(2, slice_idx)
                fix = ants.get_mask(fix, low_thresh = fix_group_id, high_thresh = fix_group_id, cleanup=0)
                # if slice_idx in [161,162] and group_id in [1,11]:
                #     fix = fix_volume.slice_image(2, slice_idx-1)
                #     fix = ants.get_mask(fix, low_thresh = fix_group_id, high_thresh = fix_group_id, cleanup=0)
                    
                if is_error:
                    txfile = ants.affine_initializer( fix, mov, use_principal_axis=True,search_factor=5, radian_fraction=0.3)
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
                                         aff_metric='meansquares', write_composite_transform=False, verbose=True, grad_step = 0.05)
                shutil.copyfile(mytx['fwdtransforms'][0], tform_name)
                
                mov_result = ants.get_mask(mytx['warpedmovout'],low_thresh = diff_const, cleanup=0)
                mov_result.to_file(mov_name)
                
                mov_result = mov_result.numpy().astype(result_volume.dtype)
                result_slice[mov_result > 0] = mov_result[mov_result > 0]*fix_group_id;
                
            result_volume[slice_idx,:,:] = result_slice
    
    img_util.write_img(result_filename, np.moveaxis(result_volume, -1, -2))    
    
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
                            subShape['Points'] = flip_lr(subShape['Points'])
     
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

def run_match_template(mov_volume_filename: str, mov_ra_filename: str, mov_label_name: str,
                       fix_volume_filename: str, result_folder: str, match_tform_name: str,
                       region_group: list, group_split: list):
    # Apply to image
    # zimg and np data are in (C, D, H, W)
    zimg_mov_volume = ZImg(mov_volume_filename)
    mov_volume = zimg_mov_volume.data[0]
    logger.info(f'finished loading template data')
    zimg_mov_label = ZImg(mov_label_name)
    mov_label = zimg_mov_label.data[0]
    logger.info(f'finished loading template label')
    zimg_fix_volume = ZImg(fix_volume_filename)
    fix_volume = zimg_fix_volume.data[0]
    logger.info(f'finished loading sample data')
    
    ants_mov_volume = ants.from_numpy(np.moveaxis(mov_label[0,:,:,:],[0, 1],[-1, -2]).astype('uint8'))
    ants_fix_volume = ants.from_numpy(np.moveaxis(fix_volume[0,:,:,:],[0, 1],[-1, -2]).astype('uint8'))
    
    ants_matched_label = ants.apply_transforms(fixed=ants_fix_volume, moving=ants_mov_volume, 
                                              transformlist=match_tform_name, interpolator='nearestNeighbor')
    matched_mov = np.zeros(shape = fix_volume.shape, dtype='uint16')
    for ch in range(mov_volume.shape[0]):
        logger.info(f'applying transform to channel {ch}')
        # ants volume is in (W, H, D)
        ants_mov_volume = ants.from_numpy(np.moveaxis(mov_volume[ch,:,:,:],[0, 1],[-1, -2]).astype('uint32'))
        ants_fix_volume = ants.from_numpy(np.moveaxis(fix_volume[ch,:,:,:],[0, 1],[-1, -2]).astype('uint32'))
        
        ants_matched_mov = ants.apply_transforms(fixed=ants_fix_volume, moving=ants_mov_volume, 
                                                  transformlist=match_tform_name, interpolator='nearestNeighbor')
        matched_mov[ch,:,:,:] = np.moveaxis(ants_matched_mov.numpy(), [-1, -2], [0, 1]).astype('uint16')
    
    # Group matching
    group_matched_volume = np.zeros(shape = fix_volume.shape, dtype='uint16')   
    group_matched_label = np.zeros(shape = (fix_volume.shape[1],fix_volume.shape[2],fix_volume.shape[3]), dtype='uint8')     
    
    slice_idx = 0
    start_idx = -1
    for idx in range(len(region_group)):
        logger.info(f'Running slice {idx}')
        if len(region_group[idx]) == 0:
            continue
        if start_idx < 0:
            start_idx = idx
        for ch in range(mov_volume.shape[0]):
            group_matched_volume[ch,idx,:,:] = matched_mov[ch, slice_idx,:,:]

        new_slice = group_matched_label[idx,:,:]
        curr_slice = np.moveaxis(ants_matched_label.slice_image(2, slice_idx).numpy(), [0,1],[1,0])
        
        region_list = np.unique(curr_slice[curr_slice>0])
        for region_id in region_list:
            for group_idx, group in enumerate(region_group[idx], start=1):
                if region_id in group or region_id-10 in group:
                    if group_idx in group_split[idx]:
                        new_slice[curr_slice == region_id] = group_idx if region_id < 10 else group_idx+10
                    else:
                        new_slice[curr_slice == region_id] = group_idx
        group_matched_label[idx,:,:] = new_slice
        slice_idx += 1
        
        
    result_filename = os.path.join(result_folder, '03_matched_template_label.nim')
    img_util.write_img(result_filename, group_matched_label)   
    
    result_filename = os.path.join(result_folder, '03_matched_template_signal.nim')
    img_util.write_img(result_filename, group_matched_volume)
    
    # Apply to region annoation
    logger.info(f'applying transform to region annotation')
    ra_dict = region_annotation.read_region_annotation(mov_ra_filename)
    
    tform_3d = ants.read_transform(match_tform_name)
    tform_3d_parameters = tform_3d.parameters
    tform_2d_parameters = [tform_3d_parameters[0], 0, 0, tform_3d_parameters[0], tform_3d_parameters[9], tform_3d_parameters[10]]
    tform_2d = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=2, parameters=tform_2d_parameters)
    tform_name = os.path.join(result_folder, 'template_match_2d.mat')
    ants.write_transform(tform_2d, tform_name)
                 
    def transform_fun(input_coords: np.ndarray):
        assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
        # res = pd.DataFrame(data=input_coords[:,::-1].copy(), columns=['x', 'y'])
        res = pd.DataFrame(data=input_coords.copy(), columns=['x', 'y'])
        res = res/16.
        res = ants.apply_transforms_to_points(2, res, tform_name, whichtoinvert=[True])
        res = res * 16.
        res = res.to_numpy()
        return res
    
    for region_id_, region in ra_dict['Regions'].items():
        if region['ROI'] is not None:
            if region_id_ == 4775:
                region['ParentID'] = 1097
            
            region_id = region_id_
                
            for img_slice, sliceROIs in region['ROI']['SliceROIs'].items():
                for shape in sliceROIs:
                    for subShape in shape:
                            # Apply transform
                            subShape['Points'] = transform_fun(subShape['Points'])

    # Write Region Annotation
    result_filename = os.path.join(result_folder, '03_matched_template_annotation.reganno')

    ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict, lambda s: s + start_idx)
    region_annotation.write_region_annotation_dict(ra_dict2, result_filename)
    
    return start_idx

def run_deform_template(mov_volume_filename: str, mov_ra_filename: str, fix_volume_filename: str, result_folder: str):
    # Apply to image
    # zimg and np data are in (C, D, H, W)
    zimg_mov_volume = ZImg(mov_volume_filename)
    mov_volume = zimg_mov_volume.data[0]
    logger.info(f'finished loading template data')
    zimg_fix_volume = ZImg(fix_volume_filename)
    fix_volume = zimg_fix_volume.data[0]
    logger.info(f'finished loading sample data')
    
    matched_mov = np.zeros(shape = fix_volume.shape, dtype='uint16')
    DAPI_ch = 1
    
    ants_mov_volume = ants.from_numpy(np.moveaxis(mov_volume[DAPI_ch,:,:,:],[0, 1],[-1, -2]).astype('uint32'))
    ants_fix_volume = ants.from_numpy(np.moveaxis(fix_volume[DAPI_ch,:,:,:],[0, 1],[-1, -2]).astype('uint32'))
     
    ra_dict = region_annotation.read_region_annotation(mov_ra_filename)
    
    for slice_idx in range(ants_mov_volume.shape[-1]):
    # for slice_idx in range(10):
        logger.info(f'running slice {slice_idx}')
        ants_mov_slice = ants_mov_volume.slice_image(2, slice_idx)
        ants_fix_slice = ants_fix_volume.slice_image(2, slice_idx)
        
        if ants_fix_slice.sum() == 0:
            continue
        mytx = ants.registration(fixed=ants_fix_slice , moving=ants_mov_slice, type_of_transform='SyNAggro', 
                                 flow_sigma = 3,total_sigma=1, reg_iterations = (1000,500,250,100), grad_step=1, 
                                 write_composite_transform=True, verbose=True)
    

        for region_id_, region in ra_dict['Regions'].items():
            if region['ROI'] is not None:
                region_id = region_id_
                if slice_idx not in region['ROI']['SliceROIs']:
                    continue
                for img_slice, sliceROIs in region['ROI']['SliceROIs'].items():
                    if img_slice == slice_idx:
                        for shape in sliceROIs:
                            for subShape in shape:
                                def transform_fun(input_coords: np.ndarray, deform_filename:str):
                                    assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
        
                                    res = pd.DataFrame(data=input_coords.copy(), columns=['x', 'y'])
                                    res = res/16.
                                    res = ants.apply_transforms_to_points(2, res, deform_filename)
                                    res = res * 16.
                                    res = res.to_numpy()
        
                                    assert res.ndim == 2 and res.shape[1] == 2, res.shape
                                    return res
        
                                # Apply transform
                                logger.info(f'applying deformation to region {region_id} in slice {slice_idx}')
                                subShape['Points'] = transform_fun(subShape['Points'], mytx['invtransforms'])

    # Write Region Annotation
    result_filename = os.path.join(result_folder, '04_deformed_template_annotation.reganno')
    region_annotation.write_region_annotation_dict(ra_dict, result_filename)    

def run_deform_template_mask(mov_volume_filename: str, mov_ra_filename: str, mov_label_filename: str, 
                             fix_volume_filename: str, fix_label_filename: str, result_folder: str,
                             blockface_group: str, template_group: str, group_split: str, group_error: list = None, slice_diff:int = 0):
    # Apply to image
    # zimg and np data are in (C, D, H, W)
    zimg_mov_volume = ZImg(mov_volume_filename)
    mov_volume = zimg_mov_volume.data[0]
    logger.info(f'finished loading template data')
    zimg_mov_label = ZImg(mov_label_filename)
    mov_label = zimg_mov_label.data[0]
    logger.info(f'finished loading template data')
    zimg_fix_volume = ZImg(fix_volume_filename)
    fix_volume = zimg_fix_volume.data[0]
    logger.info(f'finished loading sample data')
    zimg_fix_label = ZImg(fix_label_filename)
    fix_label = zimg_fix_label.data[0]
    logger.info(f'finished loading sample data')
    
    matched_mov = np.zeros(shape = fix_volume.shape, dtype='uint16')
    DAPI_ch = -1
    
    ants_mov_volume = ants.from_numpy(np.moveaxis(mov_volume[DAPI_ch,:,:,:],[0, 1],[-1, -2]).astype('uint32'))
    ants_fix_volume = ants.from_numpy(np.moveaxis(fix_volume[DAPI_ch,:,:,:],[0, 1],[-1, -2]).astype('uint32'))
    ants_mov_label = ants.from_numpy(np.moveaxis(mov_label[0,:,:,:],[0, 1],[-1, -2]).astype('uint8'))
    ants_fix_label = ants.from_numpy(np.moveaxis(fix_label[0,:,:,:],[0, 1],[-1, -2]).astype('uint8'))
     
    
    ra_dict = region_annotation.read_region_annotation(mov_ra_filename)
    region_list = [385, 669]
    slice_list = [] 
    for region_id, region in ra_dict['Regions'].items():
        if region['ParentID'] not in region_list:
            continue
        slice_list.extend(list(region['ROI']['SliceROIs'].keys()))
    # for slice_idx in range(ants_mov_volume.shape[-1]):
    # for slice_idx in range(110,170):
    for slice_idx in range(130,135):
        logger.info(f'running slice {slice_idx}')
           
        if slice_idx not in slice_list:
            logger.info('skipping')
            continue
        
        ants_mov_slice = ants_mov_volume.slice_image(2, slice_idx)
        ants_fix_slice = ants_fix_volume.slice_image(2, slice_idx)
        
        ants_mov_label_slice = ants_mov_label.slice_image(2, slice_idx)
        ants_fix_label_slice = ants_fix_label.slice_image(2, slice_idx)
        
        if ants_fix_slice.sum() == 0:
            continue
        
        label_list = np.unique(ants_fix_label_slice[ants_fix_label_slice>0])        

        for label_id in label_list:
            ants_masked_mov_label = ants.get_mask(ants_mov_label_slice, low_thresh = label_id, high_thresh = label_id, cleanup = 0)
            ants_masked_fix_label = ants.get_mask(ants_fix_label_slice, low_thresh = label_id, high_thresh = label_id, cleanup = 0)    
            ants_masked_mov = ants.mask_image(ants_mov_slice, ants_masked_mov_label)
            ants_masked_fix = ants.mask_image(ants_fix_slice, ants_masked_fix_label)
            
            if sum(sum(ants_masked_mov_label)) == 0 or sum(sum(ants_masked_fix_label)) == 0 :
                continue
            
            
            # Get regions corresponding to each label
            # check if label consist of split region
            label_on_right = label_id > 10
            if label_id > 10:
                label_id -= 10
            # Read template region_group scheme
            region_list = []
            blockface_label_list = blockface_group[slice_idx][label_id-1]
            for blockface_label_id in blockface_label_list:
                if len(template_group[slice_idx-slice_diff]) < blockface_label_id:
                    break
                region_list.extend(template_group[slice_idx-slice_diff][blockface_label_id-1])
            
            if 315 not in region_list:
                continue
             
            if 315 in region_list:
                # region_list.extend([322, 985, 993, 184, 39, 972, 378, 22, 48, 836, 895, 96,1084, 909, 385, 669, 95, 894, 254 ])
                # region_list.extend([385, 669])
                region_list = [385, 669]
            
            
            # Run initial similairy transform using mask only
            init_tx = ants.registration(fixed=ants_masked_fix_label , moving=ants_masked_mov_label, type_of_transform='Similarity', 
                         reg_iterations = (1000,500,250,100), grad_step=1, write_composite_transform=False, verbose=False)

            ants_tformed_masked_mov_label = ants.apply_transforms(fixed=ants_masked_fix_label, moving = ants_masked_mov_label, 
                                                                  transformlist=init_tx['fwdtransforms'], interpolator='nearestNeighbor')

            ants_tformed_masked_mov = ants.apply_transforms(fixed=ants_masked_fix, moving = ants_masked_mov, 
                                                                  transformlist=init_tx['fwdtransforms'], interpolator='nearestNeighbor')
            
            mytx_init = ants.registration(fixed=ants_masked_fix_label , moving=ants_tformed_masked_mov_label,
                           type_of_transform='SyNOnly', flow_sigma = 2, total_sigma=2, 
                           reg_iterations = (1000,1000,500,100), grad_step=1, write_composite_transform=True, verbose=False)
            
            ants_deformed_masked_mov = ants.apply_transforms(fixed=ants_masked_fix, moving = ants_tformed_masked_mov, 
                                                                  transformlist=mytx_init['fwdtransforms'], interpolator='nearestNeighbor')
            
            # ants_eroded_masked_fix = ants.morphology(ants_masked_fix_label, 'erode', 5)
            
            mytx = ants.registration(fixed=ants_masked_fix , moving=ants_deformed_masked_mov, 
                           type_of_transform='SyNOnly', flow_sigma = 3, total_sigma = 3,
                           reg_iterations = (1000,500,250,10), grad_step=.5, write_composite_transform=True, verbose=False)
            
            def transform_fun(input_coords: np.ndarray, tform_filename:str):
                assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
                # res = pd.DataFrame(data=input_coords[:,::-1].copy(), columns=['x', 'y'])
                res = pd.DataFrame(data=input_coords.copy(), columns=['x', 'y'])
                res = res/16.
                res = ants.apply_transforms_to_points(2, res, tform_filename, whichtoinvert=[True])
                res = res * 16.
                res = res.to_numpy()

                assert res.ndim == 2 and res.shape[1] == 2, res.shape
                return res
            
            def deform_fun(input_coords: np.ndarray, deform_filename:str):
                assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape

                res = pd.DataFrame(data=input_coords.copy(), columns=['x', 'y'])
                res = res/16.
                res = ants.apply_transforms_to_points(2, res, deform_filename)
                res = res * 16.
                res = res.to_numpy()

                assert res.ndim == 2 and res.shape[1] == 2, res.shape
                return res
            


            # Apply deformation to region annotation
            for region_id, region in ra_dict['Regions'].items():
                if region_id not in region_list and region['ParentID'] not in region_list:
                    region['ROI'] = None
                    continue
                if region['ROI'] is not None:
                    if slice_idx not in region['ROI']['SliceROIs']:
                        continue
                    
                    for img_slice, sliceROIs in region['ROI']['SliceROIs'].items():
                        if img_slice == slice_idx:
                            for shape in sliceROIs:
                                for subShape in shape:
                                    mid_x = np.mean(subShape['Points'][:,0])
                                    region_on_right = get_region_side(ra_dict, slice_idx, region_id, mid_x, 0)
                                    
                                    if label_id in group_split[slice_idx]:
                                        if label_on_right != region_on_right:
                                            # shape.remove(subShape)
                                            continue
                                        
                                    # Apply transform
                                    logger.info(f'applying deformation to region {region_id} in slice {slice_idx}')
                                    subShape['Points'] = transform_fun(subShape['Points'], init_tx['fwdtransforms'])
                                    subShape['Points'] = deform_fun(subShape['Points'], mytx_init['invtransforms'])
                                    subShape['Points'] = deform_fun(subShape['Points'], mytx['invtransforms'])

    # Write Region Annotation
    result_filename = os.path.join(result_folder, '04_deformed_template_annotation_5.reganno')
    region_annotation.write_region_annotation_dict(ra_dict, result_filename)    
        
if __name__ == "__main__":
     
    hyungju_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align'
    ref_folder = os.path.join(hyungju_folder, 'all-dataset', 'Hotsauce_SMI99_VGluT2_NeuN')
    
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
    for idx in [1]:
    # for idx in range(1):
        slice_diff = 0
        result_folder = os.path.join(hyungju_folder, 'all-dataset', result_list[idx])
        
        # ---------------------------------------------------------------------------------------------------------------------------   
        # 00. Read group and split info  
        # --------------------------------------------------------------------------------------------------------------------------- 
    
        region_group_name = os.path.join(result_folder, 'region_group.txt')
        with open(region_group_name) as json_file:
            region_group = json.load(json_file)
            
        group_split_name = os.path.join(result_folder, 'group_split.txt')
        with open(group_split_name) as json_file:
            group_split = json.load(json_file)
            
        reference_group_name = os.path.join(result_folder, 'reference_group.txt')
        with open(reference_group_name) as json_file:
            reference_group = json.load(json_file)
            
        template_group_name = os.path.join(ref_folder, 'region_group.txt')
        with open(template_group_name) as json_file:
            template_group = json.load(json_file)
            
        group_error_name = os.path.join(result_folder, 'group_error.txt')
        with open(group_error_name) as json_file:
            group_error = json.load(json_file)
        
        # ---------------------------------------------------------------------------------------------------------------------------   
        # 00. Apply Rigid Scaling using reference match information
        # ---------------------------------------------------------------------------------------------------------------------------
        template_label_filename = os.path.join(ref_folder, '02_scaled_aligned_bigregion_label.nim')

        template_volume_filename = os.path.join(ref_folder, '02_scaled_aligned_signal.tiff')
        template_ra_filename = os.path.join(ref_folder, '02_scaled_aligned_annotation_subregion_layer_tagged.reganno')
        data_volume_filename = os.path.join(result_folder, '02_aligned_signal.tiff') 
        match_tform_name = os.path.join(result_folder, 'reference_match.mat')
        # slice_diff = run_match_template(template_volume_filename, template_ra_filename, template_label_filename, data_volume_filename, result_folder,
        #                                 match_tform_name, reference_group, group_split)
        
        # ---------------------------------------------------------------------------------------------------------------------------   
        # 01. Perform SyN deformation on image
        # --------------------------------------------------------------------------------------------------------------------------- 
        template_volume_filename = os.path.join(result_folder, '03_matched_template_signal.nim')
        template_ra_filename = os.path.join(result_folder, '03_matched_template_annotation.reganno')
        template_label_filename = os.path.join(result_folder, '03_matched_template_label.nim')
        data_volume_filename = os.path.join(result_folder, '02_aligned_signal.tiff') 
        data_label_filename = os.path.join(result_folder, '02_aligned_label.nim') 
        if idx == 1:
            slice_diff = 4
        else:
            slice_diff = 0
    
        run_deform_template_mask(template_volume_filename, template_ra_filename, template_label_filename, 
                                  data_volume_filename, data_label_filename, result_folder,
                                  reference_group, template_group, group_split, group_error, slice_diff)
   