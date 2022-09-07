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

logger = setup_logger()

folder = '/home/hyungjujeon/hyungju/data'
ra_filename = os.path.join(folder, '10_layer_with_subregion_inference.reganno')
interpolate_ra_filename = os.path.join(folder, f'10_layer_with_subregion_inference_interpolate.reganno')

# folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN'
# ra_filename = os.path.join(folder, '09_layer_with_cutline.reganno')
# interpolate_ra_filename = os.path.join(folder, f'09_layer_with_cutline_interpolate.reganno')

read_ratio = 4
scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
depth = 162
height = 5072
width = 7020

interp_ratio = 5
ra_dict = region_annotation.read_region_annotation(ra_filename)
ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict, lambda s: s * interp_ratio)

logger.info(f'finish reading {ra_filename}')
region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
logger.info(f'finish reading masks from {ra_filename}')

#%%
parent_region_list = [315]


for region_id, slice_rois in region_to_masks.items():
    fix_mask = np.zeros(shape=(height, width), dtype=np.uint8)
    mov_mask = np.zeros(shape=(height, width), dtype=np.uint8)
    if region_id == -1 :
        continue
    
    
    if ra_dict['Regions'][region_id]['ParentID'] not in parent_region_list:
        if ra_dict['Regions'][ra_dict['Regions'][region_id]['ParentID']]['ParentID'] not in parent_region_list:
            continue
    for slice in range(depth-1):
    # for slice in range(10):
       logger.info(f'region {region_id} in slice {slice}')
       mov_slice = slice + 1
       
       if mov_slice not in slice_rois:
           continue

       if sum(sum(mov_mask)) > 0:
           fix_mask = mov_mask.copy()
       else:
           if slice not in slice_rois:
               continue
           maskps = slice_rois[slice]
           for compact_mask, x_start, y_start, _ in maskps:
               if compact_mask.sum() == 0:
                   continue
               assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
               mask = np.zeros(shape=(height, width), dtype=np.bool)
               mask[y_start:y_start + compact_mask.shape[0],x_start:x_start + compact_mask.shape[1]] = compact_mask
               mask = cv2.dilate(mask.astype(np.uint8),kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))) > 0
               fix_mask[mask] = region_id
       
       mov_mask = np.zeros(shape=(height, width), dtype=np.uint8) 
       maskps = slice_rois[mov_slice]
       for compact_mask, x_start, y_start, _ in maskps:
           if compact_mask.sum() == 0:
               continue
           assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
           mask = np.zeros(shape=(height, width), dtype=np.bool)
           mask[y_start:y_start + compact_mask.shape[0],x_start:x_start + compact_mask.shape[1]] = compact_mask
           mask = cv2.dilate(mask.astype(np.uint8),kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))) > 0
           mov_mask[mask] = region_id
           
       # Run SyN and stop halfway
       mov_img = ants.resample_image(ants.from_numpy(mov_mask.astype('uint32')), (4,4), False,0)
       fix_img = ants.resample_image(ants.from_numpy(fix_mask.astype('uint32')), (4,4), False,0)
       
       mytx = ants.registration(fixed=fix_img, moving=mov_img, type_of_transform='SyNOnly',
                      initial_transform = 'identity', grad_step=1, flow_sigma = 3, reg_iteration=(200,100,100),                         
                      write_composite_transform=False, verbose=True, syn_metric ='meansquares')
       
       #Stop in the middle (Forward)
       temp_filename = os.path.join(folder, "temp.nii.gz")
       for steps in range(1,interp_ratio):
           if sum(sum(mov_img)) > sum(sum(fix_img)):
               out_slice = (slice)*interp_ratio + steps
               next_deform = ants.image_read(mytx['invtransforms'][1])   
           else:
               out_slice = (slice+1)*interp_ratio - steps
               next_deform = ants.image_read(mytx['fwdtransforms'][0])
               
           next_deform = next_deform.apply(lambda x: x/interp_ratio * steps)
           next_deform.to_file(temp_filename)
           
           
           if sum(sum(mov_img)) > sum(sum(fix_img)):
               next_mid = ants.apply_transforms(mov_img, fix_img, temp_filename)
           else:
               next_mid = ants.apply_transforms(fix_img, mov_img, temp_filename)
               
           next_mid = ants.resample_image(next_mid, (1,1))
           # ants.image_write(next_mid, f'/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/{steps}.nii')
           
           shapes = nim_roi.label_image_2d_to_polygon_shapes(next_mid.numpy()>0)
           if len(shapes) > 0:
               if ra_dict2['Regions'][region_id]['ROI'] is None:
                   ra_dict2['Regions'][region_id]['ROI'] = {}
               if 'SliceROIs' not in ra_dict2['Regions'][region_id]['ROI']:
                   ra_dict2['Regions'][region_id]['ROI']['SliceROIs'] = {}
               if out_slice not in ra_dict2['Regions'][region_id]['ROI']['SliceROIs']:
                   ra_dict2['Regions'][region_id]['ROI']['SliceROIs'][out_slice] = shapes
               else:
                   ra_dict2['Regions'][region_id]['ROI']['SliceROIs'][out_slice].extend(shapes)
                   
ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict2, lambda coords: coords * read_ratio)
ra_dict2['VoxelSizeXInUM'] = 0.0625
ra_dict2['VoxelSizeYInUM'] = 0.0625
ra_dict2['VoxelSizeZInUM'] = 100 / interp_ratio
region_annotation.write_region_annotation_dict(ra_dict2, interpolate_ra_filename)