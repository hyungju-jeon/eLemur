import os
import sys
import json
import glob
import copy
import math
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
from skimage import measure
from time import perf_counter as pc

logger = setup_logger()

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

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

def visualize_density_map(cell_filename:str, ra_filename:str, result_filename:str, scale_ratio: int = 16):
    scale_down = 1.0 / scale_ratio
    mesh_to_cell_ratio = 16.0/scale_ratio
    
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_filename}')
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    logger.info(f'finish reading masks from {ra_filename}')
    
    region_list = [3851,3852,3853,3854,3855,6691,6692,6693,6694,6695]
    
    height = 1268
    width = 1755
    depth = 181
    
    ch_list = list(region_cell_count.keys())
    density_map = np.zeros(shape=(len(ch_list), depth, 1268, 1755), dtype=np.float)
    
    for region_id, slice_rois in region_to_masks.items():
        if region_id not in region_list:
            continue
        for img_slice, maskps in slice_rois.items():
            if img_slice not in region_cell_count['1'][region_id].keys():
                continue
            
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                ch_idx = 0
                
                for ch in ch_list:
                    if x_start > 870:
                        region_density = region_cell_count[ch][region_id][img_slice]['r_count'] / region_cell_count[ch][region_id][img_slice]['r_size']
                    else:
                        region_density = region_cell_count[ch][region_id][img_slice]['l_count'] / region_cell_count[ch][region_id][img_slice]['l_size']
                    
                    mask = np.zeros(shape=(height, width), dtype=np.float)
                    mask[y_start:y_start + compact_mask.shape[0],x_start:x_start + compact_mask.shape[1]] = compact_mask * (region_density)
                    density_map[ch_idx,img_slice, :, :] += mask
                    ch_idx += 1
    

if __name__ == "__main__":   
    hyungju_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align'
    ref_folder = os.path.join(hyungju_folder, 'all-dataset', 'Hotsauce_SMI99_VGluT2_NeuN')

    lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')
    folder = os.path.join(lemur_folder, 'Hotsauce_334A', '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
    img_filename = os.path.join(folder, 'hj_aligned', 'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    midline_filename = os.path.join(folder, 'interns_edited_results', 'sh_cut_in_half.reganno')
    
    
    result_list = ['Fig_PV_TH_NeuN', 
                   'Fig_SMI99_NeuN_VGlut2', 
                   'Garlic_SMI99_VGluT2_M2',
                   'Hotsauce_PV_TH_NeuN',
                   'Hotsauce_SMI99_VGluT2_NeuN',
                   'Icecream_PV_TH_NeuN',
                   'Icecream_SMI99_NeuN_VGlut2', 
                   'Jellybean_FOXP2_SMI32_NeuN',
                   'Jellybean_vGluT2_SMI32_vGluT1']
    folder_list = ['Fig_325AA/180918_Lemur-Fig_PV_TH_NeuN',
                   'Fig_325AA/180914_fig_SMI99_NeuN_VGlut2',
                   'Garlic_320CA/181023_Lemur-Garlic_SMI99_VGluT2_M2',
                   'Hotsauce_334A/181016_Lemur-Hotsauce_PV_TH_NeuN',
                   'Hotsauce_334A/181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN',
                   'Icecream_225BD/190221_icecream_PV_TH_NeuN',
                   'Icecream_225BD/20190218_icecream_SMI99_NeuN_VGlut2',
                   'Jellybean_289BD/20190813_jellybean_FOXP2_SMI32_NeuN',
                   'Jellybean_289BD/20190827_jellybean_vGluT2_SMI32_vGluT1']
    prefix_list = ['Lemur-F_PV_TH_NeuN',
                   'Lemur-F_SMI99_NeuN_VGlut2',
                   'Lemur-G_SMI99_VGluT2_M2',
                   'Lemur-H_PV_TH_NeuN',
                   'Lemur-H_SMI99_VGluT2_NeuN',
                   'Lemur-I_PV_TH_NeuN',
                   'Lemur-I_SMI99_VGluT2_NeuN',
                   'Lemur-J_FOXP2_SMI32_NeuN',
                   'Lemur-J_vGluT2_SMI32_vGluT1']
    
    for idx in [5,6]:
    # for idx in [3]:
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
            
        blockface_group_name = os.path.join(result_folder, 'blockface_group.txt')
        with open(blockface_group_name) as json_file:
            blockface_group = json.load(json_file)
            
        template_group_name = os.path.join(ref_folder, 'region_group.txt')
        with open(template_group_name) as json_file:
            template_group = json.load(json_file)
            
        group_flip_name = os.path.join(result_folder, 'group_flip.txt')
        with open(group_flip_name) as json_file:
            group_flip = json.load(json_file)
            
        # ---------------------------------------------------------------------------------------------------------------------------   
        # 01. Remap Cells
        # --------------------------------------------------------------------------------------------------------------------------- 
        group_filename = os.path.join(result_folder, '01_grouped_label.nim')
        tform_folder = os.path.join(result_folder, 'mov')
        czi_folder = os.path.join(lemur_folder, folder_list[idx])
        
        run_remap_cells(czi_folder, group_filename, tform_folder, result_folder, group_flip, group_split)      
        
        # ---------------------------------------------------------------------------------------------------------------------------   
        # 02. Cell Analysis
        # --------------------------------------------------------------------------------------------------------------------------- 
        cell_result = os.path.join(result_folder, '00_cell.json')
        with open(cell_result) as json_file:
            cells = json.load(json_file)
        ra_filename = os.path.join(result_folder, '04_deformed_template_annotation.reganno')
        
        run_analysis(cells, ra_filename, scale_ratio = 8)