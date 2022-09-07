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
from utils.brain_info import read_brain_info
from skimage.segmentation import watershed, expand_labels
import skimage
from tempfile import mktemp

logger = setup_logger()


def _callback(result):
    logger.info(f'finished {result}')

def _error_callback(e: BaseException):
    traceback.print_exception(type(e), e, e.__traceback__)
    raise e

def stack_2d_annotation(folder : str):
    (_, _, filenames) = next(os.walk(os.path.join(folder)))
    r = re.compile('.*bigregion2.*')
    filenames = list(filter(r.match, filenames))
    prefix = re.split('^(.*)_([0-9]+)_scene([0-9])_(.*)$', filenames[0])[1]
    slice_list = [int(re.split('^(.*)_([0-9]+)_scene([0-9])_(.*)$', fn)[2]) for fn in filenames]
    postfix = re.split('^(.*)_([0-9]+)_scene([0-9])_(.*)$', filenames[0])[4]
    
    combined_ra_filename = os.path.join(folder, f'{prefix}_combined.reganno')
    
    if os.path.exists(combined_ra_filename):
        logger.info(f'roi {combined_ra_filename} done')
        
    else:
        start_slice_and_ra_dict = []
        for slice_idx in range(max(slice_list)):
            for scene_idx in range(4):
                ra_name = os.path.join(folder, f'{prefix}_{slice_idx+1:02}_scene{scene_idx}_{postfix}')

                if os.path.exists(ra_name):
                    img_idx = slice_idx*4 + scene_idx

                    start_slice_and_ra_dict.append((img_idx, region_annotation.read_region_annotation(ra_name)))
        merged_ra = region_annotation.merge_region_annotation_dicts(start_slice_and_ra_dict)
        region_annotation.write_region_annotation_dict(merged_ra, combined_ra_filename)
        
        logger.info(f'roi {combined_ra_filename} done')
      
        
def stack_2d_image(folder : str, *, scale_ratio:int = 1):
    (_, _, filenames) = next(os.walk(os.path.join(folder)))
    r = re.compile('.*([0-9]{2})(.czi)$')
    filenames = list(filter(r.match, filenames))
    r = re.compile('^(?!.*pt).*$')
    filenames = list(filter(r.match, filenames))
    
    prefix = re.split('^(.*)(_[0-9]+).czi$', filenames[0])[1]
    slice_list = [int(re.split('^(.*)_([0-9]+).czi$', fn)[2]) for fn in filenames]
    slice_list = np.sort(slice_list)
    
    combined_nim_filename = os.path.join(folder, f'{prefix}_combined.nim')

    img_width = -1
    img_height = -1
    num_slice = 0
    
    for czi_name in filenames:
        czi_filename = os.path.join(folder, czi_name)
        czi_img_info = ZImg.readImgInfos(czi_filename)
        
        for scene_idx in range(len(czi_img_info)):
            if czi_img_info[scene_idx].height>img_height:
                logger.info(f'max height at {czi_name} in scene {scene_idx}')
            img_height = max(czi_img_info[scene_idx].height, img_height)
            img_width = max(czi_img_info[scene_idx].width, img_width)
            num_slice += 1
            
    img_height = int(np.ceil(img_height / scale_ratio))
    img_width = int(np.ceil(img_width / scale_ratio))
    
    # num_slice = max(slice_list)*4
    combined_nim = np.zeros((5, num_slice, img_height, img_width), dtype='uint16')
    
    slice_idx = 0
    for file_idx in slice_list:
        czi_filename = os.path.join(folder, f'{prefix}_{file_idx:02}.czi')
        logger.info(f'Running {czi_filename}')
        czi_img_info = ZImg.readImgInfos(czi_filename)
        
        for scene_idx in range(len(czi_img_info)):
            # slice_idx = (file_idx-1)*4 + scene_idx
            imgInfo = ZImg(czi_filename, region=ZImgRegion(), scene=scene_idx, xRatio=scale_ratio, yRatio=scale_ratio)
            img = imgInfo.data[0][:,0,:,:]
            
            width_pad = combined_nim.shape[-2]-img.shape[-2]
            height_pad = combined_nim.shape[-1]-img.shape[-1]
            img = np.pad(img, ((0,0), (0, width_pad), (0, 0))) 
            img = np.pad(img, ((0,0), (0, 0), (0, height_pad)))
            
            combined_nim[:,slice_idx,:,:] = img
            slice_idx += 1
            
    img_util.write_img(combined_nim_filename, combined_nim)



def stack_czi_files(result_filename: str, tform_folder: str, reflabel_name: str, movlabel_name: str, czi_folder : str, lemur_prefix : str, *, group_split: list = None, group_flip: list = None, group_error: list = None, czi_filename : str = None, run_shading_correction: bool = False, brain_info : dict = None): 
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
        
    for slice_idx in range(130,fix_volume.shape[0]):
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
            
            backgroun_filled = tformed_slice[ch].numpy()
            backgroun_filled[is_background] = background_level
            tformed_slice[ch] = ants.from_numpy(backgroun_filled)
            
        if combined_slice == None:
            combined_slice.append(np.zeros(combined_slice[0].shape, 'uint16'))
        else:
            combined_slice.append(ants.merge_channels(tformed_slice).numpy().astype('uint16'))
        # print(combined_slice[slice_idx].shape)
        
    combined_volume = np.stack(combined_slice, axis=0)
    combined_volume = np.moveaxis(combined_volume, -1, 0)
    combined_volume = np.moveaxis(combined_volume, -1, -2)
    
    img_util.write_img(result_filename, combined_volume)
 
    
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
    # for idx in range(1,2):
    for idx in [6]:
        result_folder = os.path.join(hyungju_folder, 'all-dataset', result_list[idx])
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
        # 00. Read info file (if exist)
        # --------------------------------------------------------------------------------------------------------------------------- 

        brain_info_name = os.path.join(czi_folder, 'info.txt')
        brain_info = None
        if os.path.exists(brain_info_name):
            brain_info = read_brain_info(brain_info_name)
        
        # ---------------------------------------------------------------------------------------------------------------------------   
        # 01. Run shading correction
        # --------------------------------------------------------------------------------------------------------------------------- 

        brain_info_name = os.path.join(czi_folder, 'info.txt')
        brain_info = None
        if os.path.exists(brain_info_name):
            brain_info = read_brain_info(brain_info_name)  shading-corrected
        # ---------------------------------------------------------------------------------------------------------------------------   
        # 02. Stack annotation
        # --------------------------------------------------------------------------------------------------------------------------- 
        # reflabel_filename = os.path.join(hyungju_folder, 'all-dataset', 'Hotsauce_blockface_outline_grouped_fix_interpolated_mirror.tiff')
        reflabel_filename = os.path.join(result_folder, '00_matched_reference.tif')
        stacked_label_filename = os.path.join(result_folder, '00_stacked_label.nim')
        result_filename = os.path.join(result_folder, '00_matched_reference.nim')
    
        # export_matching_reference(reflabel_filename, stacked_label_filename, result_filename, result_folder, reference_group,  reference_split)
    
