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
import meshio

from zimg import *
from utils import io
from utils import img_util
from utils import nim_roi
from utils import region_annotation
from utils.logger import setup_logger
from utils.brain_info import read_brain_info
from utils import shading_correction
from utils.lemur_ontology import *
from skimage import measure
from skimage.segmentation import (flood_fill, expand_labels)
from skimage.morphology import (opening, closing, square, disk, dilation, erosion, binary_opening, binary_closing)
from scipy.ndimage.morphology import binary_fill_holes

logger = setup_logger()


def _callback(result):
    logger.info(f'finished {result}')


def _error_callback(e: BaseException):
    traceback.print_exception(type(e), e, e.__traceback__)
    raise e


def search_all_child(ra_dict:dict, parent_list:list):
    child_list = []
    for region_id, region_props in ra_dict['Regions'].items():
        if region_props['ParentID'] in parent_list:
            child_list.append(region_id)
            child_list += search_all_child(ra_dict, [region_id])
    return child_list


def convert_mesh_to_mask(mesh_folder: str, result_folder: str = None):
    # List mesh files
    (_, path_name, file_list) = next(os.walk(mesh_folder))
    file_list = [fn for fn in file_list if '.obj' in fn and ',' not in fn]

    for region_name in file_list:
        mesh_filename = os.path.join(mesh_folder, region_name)
        result_filename = os.path.join(result_folder, f'{region_name[:-4]}.nim')
        if os.path.exists(result_filename):
            logger.info(f'{region_name} already exist')
        else:
            # Open using ZMesh
            logger.info(f'Currently processing region {region_name}')
            msh = ZMesh(mesh_filename)
            # mesh_id = get_id(ontology, meshname)

            # Rescale and center vertices
            mesh_vertices = msh.vertices
            mesh_vertices[:, 2] /= 160
            mesh_vertices[:, 2] += 92.
            mesh_vertices[:, 0] /= 16
            mesh_vertices[:, 1] /= 16
            msh.vertices = mesh_vertices

            # Voxelize and clear mesh file
            img = msh.toLabelImg(width=1755, height=1268, depth=180)

            for slice_idx in range(img.data[0][0,:,:,:].shape[0]):
                slice_img = img.data[0][0,slice_idx,:,:].copy()
                if np.sum(slice_img) == 0:
                    continue
                filled_slice = binary_fill_holes(slice_img).astype(int)
                holes = opening(filled_slice - slice_img, square(3))
                filled_slice = filled_slice - holes
                slice_img = opening(filled_slice, square(5))
                slice_img = closing(slice_img, square(5))
                props = measure.regionprops_table(measure.label(slice_img), properties=('area', 'coords'))
                noise_artfct = [i for i, x in enumerate(props['area']) if x < 300]
                if len(noise_artfct) > 0:
                    noise_voxel = np.concatenate(props['coords'][noise_artfct])
                    noise_voxel = [np.ravel_multi_index(x, slice_img.shape) for x in noise_voxel]
                    slice_img[np.unravel_index(noise_voxel, slice_img.shape)] = 0
                img.data[0][0,slice_idx,:,:] = slice_img
            img.save(result_filename)


def refine_3d_isotropic_cortex_layer(mesh_folder: str, result_filename: str = None, ra_filename: str=None):
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * 1/16.)
    # region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)

    l1_filename = os.path.join(mesh_folder, 'Layer1.nim')
    l2_filename = os.path.join(mesh_folder, 'Layer2&3.nim')
    l4_filename = os.path.join(mesh_folder, 'Layer4.nim')
    l5_filename = os.path.join(mesh_folder, 'Layer5.nim')
    l6_filename = os.path.join(mesh_folder, 'Layer6.nim')
    cortex_filename = os.path.join(mesh_folder, 'Isocortex.nim')

    # Load voxelized mesh
    l1_zimg = ZImg(l1_filename)
    l2_zimg = ZImg(l2_filename)
    l4_zimg = ZImg(l4_filename)
    l5_zimg = ZImg(l5_filename)
    l6_zimg = ZImg(l6_filename)
    cortex_zimg = ZImg(cortex_filename)
    l1_mask = l1_zimg.data[0][0,:,:,:].copy()
    l2_mask = l2_zimg.data[0][0,:,:,:].copy()
    l4_mask = l4_zimg.data[0][0,:,:,:].copy()
    l5_mask = l5_zimg.data[0][0,:,:,:].copy()
    l6_mask = l6_zimg.data[0][0,:,:,:].copy()
    cortex_mask = cortex_zimg.data[0][0,:,:,:].copy()

    for region_id, region_props in ra_dict['Regions'].items():
        if region_id not in [3155, 3156, 3157, 3158, 3159]:
            region_props['ROI'] = None
    full_ra_dict = copy.deepcopy(ra_dict)

    for slice_idx in range(cortex_mask.shape[0]):
        logger.info(f'Running slice {slice_idx}')
        l1_slice = l1_mask[slice_idx, :, :]
        l2_slice = l2_mask[slice_idx, :, :]
        l4_slice = l4_mask[slice_idx, :, :]
        l5_slice = l5_mask[slice_idx, :, :]
        l6_slice = l6_mask[slice_idx, :, :]
        cortex_slice = cortex_mask[slice_idx, :, :]

        if slice_idx in range(1070, 1120):
            temp = ((cortex_slice + l1_slice + l2_slice) > 0) * 1.
            temp = np.pad(array=temp, pad_width=20)
            temp = cv2.morphologyEx(cv2.morphologyEx(temp, cv2.MORPH_CLOSE, disk(10)), cv2.MORPH_OPEN, disk(15))
            cortex_slice = ((temp[20:-20, 20:-20] + cortex_slice) > 0) * 1.
        if slice_idx in range(1076, 1094):
            # Create updated cortex
            slice_img = l1_slice * 3159 + l2_slice * 3158 + l4_slice * 3157 + l5_slice * 3156 + l6_slice * 3155
            slice_img[slice_img > 3160] = 0
            slice_img[cortex_slice == 0] = 0
            guide_slice = l1_mask[1094, :, :]*3159 + l2_mask[1094, :, :]*3158 + l4_mask[1094, :, :]*3157 \
                          + l5_mask[1094, :, :]*3156 + l6_mask[1094, :, :]*3155
            guide_slice[cortex_slice == 0] = 0
            guide_slice[guide_slice > 3160] = 0
            guide_slice = expand_labels(guide_slice, 2)
            for region_id in [3155, 3156, 3159]:
                mask = (guide_slice == region_id)
                slice_img[(guide_slice == region_id) & (slice_img == 0)] = region_id
        else:
            slice_img = l1_slice * 3159 + l2_slice * 3158 + l4_slice * 3157 + l5_slice * 3156 + l6_slice * 3155
            slice_img[slice_img > 3160] = 0
            slice_img[cortex_slice == 0] = 0

        slice_img = expand_labels(slice_img, 20)
        slice_img[cortex_slice == 0] = 0
        slice_img = np.roll(slice_img, 20, axis=0)
        l1_mask[slice_idx, :, :] = slice_img == 3159
        l2_mask[slice_idx, :, :] = slice_img == 3158
        l4_mask[slice_idx, :, :] = slice_img == 3157
        l5_mask[slice_idx, :, :] = slice_img == 3156
        l6_mask[slice_idx, :, :] = slice_img == 3155
        cortex_slice = np.roll(cortex_slice, 20, axis=0)
        cortex_mask[slice_idx, :, :] = cortex_slice

        for region_id in [3155, 3156, 3157, 3158, 3159]:
            region_spline = nim_roi.mask_2d_to_polygon_shapes(slice_img == region_id)
            if slice_idx / 10 == slice_idx // 10:
                if ra_dict['Regions'][region_id]['ROI'] is None:
                    ra_dict['Regions'][region_id]['ROI'] = {}
                if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                    ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                if slice_idx // 10 in ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                    ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx // 10] = region_spline.copy()
            if full_ra_dict['Regions'][region_id]['ROI'] is None:
                full_ra_dict['Regions'][region_id]['ROI'] = {}
            if 'SliceROIs' not in full_ra_dict['Regions'][region_id]['ROI']:
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
            if slice_idx // 10 in full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
            full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()

    for slice_idx in range(3):
        for region_id in [3155, 3156, 3157, 3158, 3159]:
            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = ra_dict['Regions'][region_id]['ROI']['SliceROIs'][
                3].copy()
    for slice_idx in range(30):
        for region_id in [3155, 3156, 3157, 3158, 3159]:
            if 30 in full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][
                    30].copy()

    img_util.write_img(os.path.join(mesh_folder, 'resolved', 'Layer1.nim'), l1_mask)
    img_util.write_img(os.path.join(mesh_folder, 'resolved', 'Layer2&3.nim'), l2_mask)
    img_util.write_img(os.path.join(mesh_folder, 'resolved', 'Layer4.nim'), l4_mask)
    img_util.write_img(os.path.join(mesh_folder, 'resolved', 'Layer5.nim'), l5_mask)
    img_util.write_img(os.path.join(mesh_folder, 'resolved', 'Layer6.nim'), l6_mask)
    img_util.write_img(os.path.join(mesh_folder, 'resolved', 'Isocortex.nim'), cortex_mask)

    region_annotation.write_region_annotation_dict(ra_dict, result_filename)
    region_annotation.write_region_annotation_dict(full_ra_dict, result_filename[:-8] + '_interp.reganno')


def refine_3d_isotropic_cortex_subregion(mesh_folder: str, result_filename: str = None, ra_filename: str=None):
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()

    # Load layer Annotation
    ra_dict = region_annotation.read_region_annotation(ra_filename)

    # Load refined mask
    refined_mask_folder = os.path.join(mesh_folder, 'resolved')
    l1_filename = os.path.join(refined_mask_folder, 'Layer1.nim')
    l2_filename = os.path.join(refined_mask_folder, 'Layer2&3.nim')
    l4_filename = os.path.join(refined_mask_folder, 'Layer4.nim')
    l5_filename = os.path.join(refined_mask_folder, 'Layer5.nim')
    l6_filename = os.path.join(refined_mask_folder, 'Layer6.nim')
    cortex_filename = os.path.join(refined_mask_folder, 'Isocortex.nim')
    # Load voxelized mesh
    l1_zimg = ZImg(l1_filename)
    l2_zimg = ZImg(l2_filename)
    l4_zimg = ZImg(l4_filename)
    l5_zimg = ZImg(l5_filename)
    l6_zimg = ZImg(l6_filename)
    cortex_zimg = ZImg(cortex_filename)
    l1_mask = l1_zimg.data[0][0,:,:,:].copy()
    l2_mask = l2_zimg.data[0][0,:,:,:].copy()
    l4_mask = l4_zimg.data[0][0,:,:,:].copy()
    l5_mask = l5_zimg.data[0][0,:,:,:].copy()
    l6_mask = l6_zimg.data[0][0,:,:,:].copy()
    cortex_mask = cortex_zimg.data[0][0,:,:,:].copy()

    width = cortex_mask.shape[2]
    height = cortex_mask.shape[1]
    depth = cortex_mask.shape[0]

    # Load all area mask
    (_, path_name, file_list) = next(os.walk(mesh_folder))
    file_list = [fn for fn in file_list if 'Area' in fn]

    region_mask = np.zeros((1, depth, height, width), dtype='uint16')
    merged_region = np.zeros((1, depth, height, width), dtype='uint16')
    region_list = []
    for region_name in file_list:
        region_zimg = ZImg(os.path.join(mesh_folder, region_name))
        region_id = get_id_from_ontology(ontology, region_name[:-4])
        region_list.append(region_id)
        region_img = region_zimg.data[0][0, :, :, :].copy()

        merged_region[0, :, :, :] = merged_region[0, :, :, :] + region_img * region_id
        region_mask += region_img
    merged_region[region_mask > 1] = 0

    for region_id, region_props in ra_dict['Regions'].items():
        if region_id not in region_list:
            region_props['ROI'] = None
    full_ra_dict = copy.deepcopy(ra_dict)

    for slice_idx in range(cortex_mask.shape[0]):
        logger.info(f'Refining subregions in slice {slice_idx}')
        l1_slice = l1_mask[slice_idx, :, :]
        l2_slice = l2_mask[slice_idx, :, :]
        l4_slice = l4_mask[slice_idx, :, :]
        l5_slice = l5_mask[slice_idx, :, :]
        l6_slice = l6_mask[slice_idx, :, :]
        cortex_slice = cortex_mask[slice_idx, :, :]
        layer_slice = l1_slice * 3159 + l2_slice * 3158 + l4_slice * 3157 + l5_slice * 3156 + l6_slice * 3155

        # Expand region label and crop with cortex mask
        slice_img = merged_region[0, slice_idx, :, :]
        slice_img = np.roll(slice_img, 20, axis=0)
        slice_img[cortex_slice == 0] = 0

        slice_img = expand_labels(slice_img, distance=20)
        slice_img[layer_slice == 0] = 0
        for region_id in np.unique(slice_img):
            if region_id > 0:
                temp_mask = (slice_img == region_id)*1.
                temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, disk(7))
                slice_img[slice_img == region_id] = temp_mask[slice_img == region_id] * region_id
        slice_img = expand_labels(slice_img, distance=20)
        slice_img[layer_slice == 0] = 0
        merged_region[0,slice_idx,:,:] = slice_img

        for region_id in region_list:
            region_spline = nim_roi.mask_2d_to_polygon_shapes(slice_img == region_id)
            if slice_idx / 10 == slice_idx // 10:
                if ra_dict['Regions'][region_id]['ROI'] is None:
                    ra_dict['Regions'][region_id]['ROI'] = {}
                if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                    ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                if slice_idx // 10 in ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                    ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx // 10] = region_spline.copy()
            if full_ra_dict['Regions'][region_id]['ROI'] is None:
                full_ra_dict['Regions'][region_id]['ROI'] = {}
            if 'SliceROIs' not in full_ra_dict['Regions'][region_id]['ROI']:
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
            if slice_idx // 10 in full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
            full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()


    for slice_idx in range(3):
        for region_id in region_list:
            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = ra_dict['Regions'][region_id]['ROI']['SliceROIs'][
                3].copy()
    for slice_idx in range(30):
        for region_id in region_list:
            if 30 in full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][
                    30].copy()


    region_annotation.write_region_annotation_dict(ra_dict, result_filename)
    region_annotation.write_region_annotation_dict(full_ra_dict, result_filename[:-8] + '_interp.reganno')

    for region_name in file_list:
        region_id = get_id_from_ontology(ontology, region_name[:-4])
        nim_filename = os.path.join(refined_mask_folder, region_name)
        img_util.write_img(nim_filename, (merged_region[0,:,:,:] == region_id).astype('uint8'))


def refine_3d_isotropic_all_subregion(mask_folder: str, result_filename: str = None, ra_filename: str=None):
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()

    # Load layer Annotation
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    for region_id, region_props in ra_dict['Regions'].items():
        region_props['ROI'] = None
    full_ra_dict = copy.deepcopy(ra_dict)

    # Load refined mask
    refined_mask_folder = os.path.join(mask_folder, 'resolved')
    cortex_filename = os.path.join(refined_mask_folder, 'Isocortex.nim')
    # Load voxelized mesh
    cortex_zimg = ZImg(cortex_filename)
    cortex_mask = cortex_zimg.data[0][0,:,:,:].copy()

    width = cortex_mask.shape[2]
    height = cortex_mask.shape[1]
    depth = cortex_mask.shape[0]

    # Load all area mask
    (_, path_name, file_list) = next(os.walk(mask_folder))
    file_list = [fn for fn in file_list if 'Area' not in fn and 'Layer' not in fn and '.nim' in fn]
    # file_list = [fn for fn in file_list if 'Isocortex' not in fn]
    id_list = [get_id_from_ontology(ontology, fn[:-4]) for fn in file_list]
    file_id_list = list(zip(id_list, file_list))

    # Get all the big regions that exist
    big_region_id_tuple = [(_id, _name) for _id, _name in file_id_list if get_region_from_ontology(ontology, _id)['parent_structure_id'] not in id_list]
    big_region_id_list, big_region_file_list = zip(*big_region_id_tuple)
    big_region_id_list = list(big_region_id_list)
    big_region_file_list = list(big_region_file_list)
    child_region_id_tuple = [(_id, _name) for _id, _name in file_id_list if (_id, _name) not in big_region_id_tuple]
    child_region_id_list, child_region_file_list = zip(*child_region_id_tuple)
    child_region_id_list = list(child_region_id_list)
    child_region_file_list = list(child_region_file_list)
    # Fix big regions
    big_region_label = merge_and_refine_mesh(big_region_file_list, mask_folder, ontology, expand_px=30, refine_px=5)

    for slice_idx in range(depth):
        logger.info(f'Refining subregions in slice {slice_idx}')
        slice_img = big_region_label[slice_idx, :, :]
        slice_img[cortex_mask[slice_idx, :, :] > 0] = 0
        region_list = np.unique(slice_img)
        region_list = [idx for idx in region_list if idx > 10]
        for region_id in region_list:
            region_spline = nim_roi.mask_2d_to_polygon_shapes(slice_img == region_id)
            if slice_idx / 10 == slice_idx // 10:
                if ra_dict['Regions'][region_id]['ROI'] is None:
                    ra_dict['Regions'][region_id]['ROI'] = {}
                if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                    ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                if slice_idx // 10 in ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                    ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx // 10] = region_spline.copy()
            if full_ra_dict['Regions'][region_id]['ROI'] is None:
                full_ra_dict['Regions'][region_id]['ROI'] = {}
            if 'SliceROIs' not in full_ra_dict['Regions'][region_id]['ROI']:
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
            if slice_idx // 10 in full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
            full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()

    # Fix children structures
    parent_list = [get_region_from_ontology(ontology,_id)['parent_structure_id'] for _id,_name in child_region_id_tuple]
    for parent_id in np.unique(parent_list) :
        logger.info(f'Refining children of {parent_id}')
        parent_mask = (big_region_label == parent_id) * 1.
        children_list = [child_region_file_list[idx] for idx in np.where(np.array(parent_list) == parent_id)[0]]
        child_label = merge_and_refine_mesh(children_list, mask_folder, ontology, parent_mask=parent_mask,
                                            expand_px=5)
        for slice_idx in range(depth):
            logger.info(f'Refining subregions in slice {slice_idx}')
            slice_img = child_label[slice_idx, :, :]
            region_list = np.unique(slice_img)
            region_list = [idx for idx in region_list if idx > 10]
            for region_id in region_list:
                region_spline = nim_roi.mask_2d_to_polygon_shapes(slice_img == region_id)
                if slice_idx / 10 == slice_idx // 10:
                    if ra_dict['Regions'][region_id]['ROI'] is None:
                        ra_dict['Regions'][region_id]['ROI'] = {}
                    if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                        ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                    if slice_idx // 10 in ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                        ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
                    ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx // 10] = region_spline.copy()
                if full_ra_dict['Regions'][region_id]['ROI'] is None:
                    full_ra_dict['Regions'][region_id]['ROI'] = {}
                if 'SliceROIs' not in full_ra_dict['Regions'][region_id]['ROI']:
                    full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                if slice_idx // 10 in full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                    full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()

    for slice_idx in range(3):
        for region_id in region_list:
            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = ra_dict['Regions'][region_id]['ROI']['SliceROIs'][
                3].copy()
    for slice_idx in range(30):
        for region_id in region_list:
            if 30 in full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][
                    30].copy()

    # If a parent structure does not exist, create one


    region_annotation.write_region_annotation_dict(ra_dict, result_filename)
    region_annotation.write_region_annotation_dict(full_ra_dict, result_filename[:-8] + '_interp.reganno')

    file_list = [fn for fn in file_list if 'Isocortex' not in fn]
    for region_name in file_list:
        region_id = get_id_from_ontology(ontology, region_name[:-4])
        nim_filename = os.path.join(refined_mask_folder, region_name)
        img_util.write_img(nim_filename, (big_region_label == region_id).astype('uint8'))


def merge_isotropic_subregion_layer(mask_folder: str, ra_filename: str, result_filename: str=None):
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()

    # Load layer Annotation
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    for region_id, region_props in ra_dict['Regions'].items():
        region_props['ROI'] = None
    full_ra_dict = copy.deepcopy(ra_dict)

    # Load all subregion and layer mask
    (_, path_name, file_list) = next(os.walk(mask_folder))
    area_list = [fn for fn in file_list if 'Area' in fn]
    layer_list = [fn for fn in file_list if 'Layer' in fn]
    cortex_filename = os.path.join(mask_folder, 'Isocortex.nim')
    cortex_zimg = ZImg(cortex_filename)
    cortex_mask = cortex_zimg.data[0][0,:,:,:].copy()

    width = cortex_mask.shape[2]
    height = cortex_mask.shape[1]
    depth = cortex_mask.shape[0]

    # Create merged image
    merged_region = np.zeros((depth, height, width), dtype='uint16')
    region_list = []
    for region_name in area_list:
        region_zimg = ZImg(os.path.join(mask_folder, region_name))
        region_id = get_id_from_ontology(ontology, region_name[:-4])
        region_list.append(region_id)
        region_img = region_zimg.data[0][0, :, :, :].copy()

        merged_region = merged_region + region_img * region_id

    merged_layer = np.zeros((depth, height, width), dtype='uint16')
    region_list = []
    for region_name in layer_list:
        region_zimg = ZImg(os.path.join(mask_folder, region_name))
        region_id = get_id_from_ontology(ontology, region_name[:-4])
        region_list.append(region_id)
        region_img = region_zimg.data[0][0, :, :, :].copy()

        merged_layer = merged_layer + region_img * region_id

    merged_region[merged_region>0] = merged_region[merged_region>0]*10. + (3160 - merged_layer[merged_region>0])

    for slice_idx in range(cortex_mask.shape[0]):
        logger.info(f'Refining subregions in slice {slice_idx}')
        slice_img = merged_region[slice_idx, :, :]
        region_list = np.unique(slice_img)
        region_list = [idx for idx in region_list if idx > 10]
        for region_id in region_list:
            region_spline = nim_roi.mask_2d_to_polygon_shapes(slice_img == region_id)
            if slice_idx / 10 == slice_idx // 10:
                if ra_dict['Regions'][region_id]['ROI'] is None:
                    ra_dict['Regions'][region_id]['ROI'] = {}
                if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                    ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                if slice_idx // 10 in ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                    ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx // 10] = region_spline.copy()
            if full_ra_dict['Regions'][region_id]['ROI'] is None:
                full_ra_dict['Regions'][region_id]['ROI'] = {}
            if 'SliceROIs' not in full_ra_dict['Regions'][region_id]['ROI']:
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
            if slice_idx // 10 in full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
            full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()


    for slice_idx in range(3):
        for region_id in region_list:
            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = ra_dict['Regions'][region_id]['ROI']['SliceROIs'][
                3].copy()
    for slice_idx in range(30):
        for region_id in region_list:
            if 30 in full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][
                    30].copy()

    region_annotation.write_region_annotation_dict(ra_dict, result_filename)
    region_annotation.write_region_annotation_dict(full_ra_dict, result_filename[:-8] + '_interp.reganno')

    for region_name in area_list:
        for layer_name in ['Layer1', 'Layer2/3', 'Layer4', 'Layer5', 'Layer6']:
            region_id = get_id_from_ontology(ontology, region_name[:-4] + ', ' + layer_name)
            if layer_name == 'Layer2/3':
                nim_filename = os.path.join(mask_folder, region_name[:-4] + ', Layer2_3.nim')
            else:
                nim_filename = os.path.join(mask_folder, region_name[:-4]+', '+layer_name+'.nim')
            img_util.write_img(nim_filename, (merged_region[:,:,:] == region_id).astype('uint8'))


def refine_3d_annotation_final(ra_filename: str, result_filename: str, resolved_folder: str):
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()

    # Load layer Annotation
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    height = 1268
    width = 1755

    cortex_region_list = search_all_child(ra_dict, [315])
    cortex_region_list = [id for id in cortex_region_list if id not in [395,394,393,392,391,9721,9722,9723,9724,9725]]

    for slice_idx in range(180):
        annotation_mask = np.zeros(shape=(1268, 1755), dtype='uint16')
        for region_id, slice_rois in region_to_masks.items():
            if slice_idx in slice_rois:
                maskps = slice_rois[slice_idx]
                for compact_mask, x_start, y_start, _ in maskps:
                    if compact_mask.sum() == 0:
                        continue
                    assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                    mask = np.zeros(shape=(height, width),  dtype=np.bool)
                    mask[y_start:y_start + compact_mask.shape[0],
                    x_start:x_start + compact_mask.shape[1]] = compact_mask
                    annotation_mask[mask] = region_id
        region_mask = cv2.morphologyEx((annotation_mask>0)*1., cv2.MORPH_CLOSE, disk(10))
        region_mask = binary_fill_holes(region_mask)
        # Expand all slightly (5 px)
        cortex_mask = np.zeros(shape=(1268, 1755), dtype='uint16')
        for cortex_id in cortex_region_list:
            cortex_mask[annotation_mask == cortex_id] = 1
        cortex_mask = cv2.morphologyEx(cortex_mask, cv2.MORPH_CLOSE, disk(2))
        annotation_mask[cortex_mask > 0] = 0
        region_mask[cortex_mask > 0] = 0
        annotation_mask = expand_labels(annotation_mask, distance=5)
        annotation_mask[region_mask == 0] = 0

        # Expand fibertract to
        expand_region_list = [1009, 803, 549, 3111]
        expand_mask = np.zeros(shape=(1268, 1755), dtype='uint16')
        for expand_id in expand_region_list:
            expand_mask[annotation_mask == expand_id] = expand_id
        expand_mask = expand_labels(expand_mask, distance=75)
        expand_mask[region_mask == 0] = 0
        annotation_mask[annotation_mask == 0] = expand_mask[annotation_mask == 0]

        region_list = [id for id in np.unique(annotation_mask) if id > 0]
        for region_id in region_list:
            child_list = search_all_child(ra_dict, [region_id])
            region_mask = annotation_mask == region_id
            for child_id in child_list:
                region_mask = region_mask | (annotation_mask == child_id)
            region_spline = nim_roi.mask_2d_to_polygon_shapes(region_mask)
            if ra_dict['Regions'][region_id]['ROI'] is None:
                ra_dict['Regions'][region_id]['ROI'] = {}
            if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
            if slice_idx // 10 in ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
                ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx)
            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()

            # if full_ra_dict['Regions'][region_id]['ROI'] is None:
            #     full_ra_dict['Regions'][region_id]['ROI'] = {}
            # if 'SliceROIs' not in full_ra_dict['Regions'][region_id]['ROI']:
            #     full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
            # if slice_idx // 10 in full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].keys():
            #     full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx // 10)
            # full_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()
    region_annotation.write_region_annotation_dict(ra_dict, result_filename)


def refine_3d_cortex_subregion(mesh_folder: str, result_filename: str = None, ra_filename: str=None):
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()

    # Load layer Annotation
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)

    merged_layer = np.zeros((1, 180, 1268, 1755), dtype='uint16')
    merged_cortex = np.zeros((1, 180, 1268, 1755), dtype='uint16')
    for region_id in [3155, 3156, 3157, 3158, 3159]:
        for slice_idx, maskps in region_to_masks[region_id].items():
            mask = np.zeros(shape=(1268, 1755), dtype='uint16')
            for compact_mask, x_start, y_start, _ in maskps:
                mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] = mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] | compact_mask

            merged_slice = merged_layer[0, slice_idx, :, :]
            merged_slice[mask>0] = mask[mask>0] * region_id
            merged_layer[0, slice_idx, :, :] = merged_slice
    merged_cortex = (merged_layer > 0)*1.
    for slice_idx in range(merged_cortex.shape[1]):
        merged_cortex[0, slice_idx, :, :] = closing(merged_cortex[0, slice_idx, :, :], disk(3))

    # Load all area mesh
    (_, path_name, file_list) = next(os.walk(mesh_folder))
    file_list = [fn for fn in file_list if 'Area' in fn]

    region_mask = np.zeros((1, 180, 1268, 1755), dtype='uint16')
    merged_region = np.zeros((1, 180, 1268, 1755), dtype='uint16')
    region_list = []
    for region_name in file_list:
        region_zimg = ZImg(os.path.join(mesh_folder, region_name))
        region_id = get_id_from_ontology(ontology, region_name[:-4])
        region_list.append(region_id)
        region_img = region_zimg.data[0][0, :, :, :].copy()

        merged_region[0, :, :, :] = merged_region[0, :, :, :] + region_img * region_id
        region_mask += region_img
    merged_region[region_mask > 1] = 0

    # Expand region label and crop with cortex mask
    for slice_idx in range(merged_region.shape[1]):
        logger.info(f'Refining slice {slice_idx}')
        slice_img = merged_region[0, slice_idx, :, :]
        slice_img = opening(slice_img, disk(3))
        slice_img = expand_labels(slice_img, distance=20)
        slice_img[merged_cortex[0, slice_idx, :, :] == 0] = 0
        for region_id in np.unique(slice_img):
            if region_id > 0:
                temp_mask = slice_img == region_id
                temp_mask = binary_opening(temp_mask, disk(10))
                slice_img[slice_img == region_id] = temp_mask[slice_img == region_id] * region_id
        slice_img = expand_labels(slice_img, distance=20)
        slice_img[merged_cortex[0, slice_idx, :, :] == 0] = 0
        for region_id in np.unique(slice_img):
            if region_id > 0:
                temp_mask = slice_img == region_id
                temp_mask = binary_opening(temp_mask, disk(10))
                slice_img[slice_img == region_id] = temp_mask[slice_img == region_id] * region_id
        slice_img = expand_labels(slice_img, distance=20)
        slice_img[merged_cortex[0, slice_idx, :, :] == 0] = 0
        merged_region[0,slice_idx,:,:] = slice_img

    for slice_idx in range(merged_region.shape[1]):
        logger.info(f'Saving slice {slice_idx}')
        for region_id, region in ra_dict['Regions'].items():
            if region_id in region_list:
                slice_img = merged_region[0, slice_idx, :, :]
                layer_img = merged_layer[0, slice_idx, :, :]

                region_spline = nim_roi.mask_2d_to_polygon_shapes(slice_img == region_id)
                if region['ROI'] is None:
                    region['ROI'] = {}
                if 'SliceROIs' not in region['ROI']:
                    region['ROI']['SliceROIs'] = {}
                region['ROI']['SliceROIs'][slice_idx] = region_spline.copy()

            else:
                region['ROI'] = None

    # ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * 16.)
    region_annotation.write_region_annotation_dict(ra_dict, result_filename)


def merge_subregion_layer(layer_ra_filename: str, subregion_ra_filename: str, result_filename: str=None):
    # Load layer and region Annotation
    layer_ra_dict = region_annotation.read_region_annotation(layer_ra_filename)
    region_ra_dict = region_annotation.read_region_annotation(subregion_ra_filename)
    layer_to_masks = region_annotation.convert_region_annotation_dict_to_masks(layer_ra_dict)
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(region_ra_dict)

    # Create merged image
    merged_region = np.zeros((1, 180, 1268, 1755), dtype='uint16')
    for region_id in [3155, 3156, 3157, 3158, 3159]:
        for slice_idx, maskps in layer_to_masks[region_id].items():
            mask = np.zeros(shape=(1268, 1755), dtype='uint16')
            for compact_mask, x_start, y_start, _ in maskps:
                mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] = mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] | compact_mask

            merged_slice = merged_region[0, slice_idx, :, :]
            merged_slice[mask>0] = mask[mask>0] * (3160 - region_id)
            merged_region[0, slice_idx, :, :] = merged_slice

    for region_id, slice_rois in region_to_masks.items():
        for slice_idx, maskps in region_to_masks[region_id].items():
            mask = np.zeros(shape=(1268, 1755), dtype='uint16')
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] = mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] | compact_mask

            merged_slice = merged_region[0, slice_idx, :, :]
            merged_slice = merged_slice + mask * region_id * 10
            merged_region[0, slice_idx, :, :] = merged_slice

    region_list = np.unique(merged_region)
    for slice_idx in range(merged_region.shape[1]):
        logger.info(f'Saving slice {slice_idx}')
        for region_id, region in region_ra_dict['Regions'].items():
            if region_id in region_list:
                slice_img = merged_region[0, slice_idx, :, :]

                region_spline = nim_roi.mask_2d_to_polygon_shapes(slice_img == region_id)
                if region['ROI'] is None:
                    region['ROI'] = {}
                if 'SliceROIs' not in region['ROI']:
                    region['ROI']['SliceROIs'] = {}
                region['ROI']['SliceROIs'][slice_idx] = region_spline.copy()
            else:
                region['ROI'] = None

    # ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * 16.)
    region_annotation.write_region_annotation_dict(region_ra_dict, result_filename)


def interp_3d_layer(ra_filename: str, result_filename: str = None):
    folder = os.path.dirname(ra_filename)
    read_ratio = 1
    scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
    depth = 180
    height = 1268
    width = 1755

    interp_ratio = 5
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict, lambda s: s * interp_ratio)

    logger.info(f'finish reading {ra_filename}')
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    logger.info(f'finish reading masks from {ra_filename}')

    #
    parent_region_list = [315]

    for region_id, slice_rois in region_to_masks.items():
        fix_mask = np.zeros(shape=(height, width), dtype=np.uint8)
        mov_mask = np.zeros(shape=(height, width), dtype=np.uint8)
        if region_id == -1:
            continue

        if region_id not in parent_region_list:
            if ra_dict['Regions'][region_id]['ParentID'] not in parent_region_list:
                if ra_dict['Regions'][ra_dict['Regions'][region_id]['ParentID']][
                    'ParentID'] not in parent_region_list:
                    continue
        for slice in range(depth - 1):
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
                    mask[y_start:y_start + compact_mask.shape[0],
                    x_start:x_start + compact_mask.shape[1]] = compact_mask
                    mask = cv2.dilate(mask.astype(np.uint8),
                                      kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))) > 0
                    fix_mask[mask] = region_id

            mov_mask = np.zeros(shape=(height, width), dtype=np.uint8)
            maskps = slice_rois[mov_slice]
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                mask = np.zeros(shape=(height, width), dtype=np.bool)
                mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] = compact_mask
                mask = cv2.dilate(mask.astype(np.uint8),
                                  kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))) > 0
                mov_mask[mask] = region_id

            # Run SyN and stop halfway
            mov_img = ants.resample_image(ants.from_numpy(mov_mask.astype('uint32')), (2, 2), False, 0)
            fix_img = ants.resample_image(ants.from_numpy(fix_mask.astype('uint32')), (2, 2), False, 0)
            # mov_img = ants.from_numpy(mov_mask.astype('uint32'))
            # fix_img = ants.from_numpy(fix_mask.astype('uint32'))

            mytx = ants.registration(fixed=fix_img, moving=mov_img, type_of_transform='SyNOnly',
                                     initial_transform='identity', grad_step=1, flow_sigma=2, total_sigma=0,
                                     reg_iteration=(100, 200, 200, 50),
                                     write_composite_transform=False, verbose=False)

            # Stop in the middle (Forward)
            temp_filename = os.path.join(folder, "temp.nii.gz")
            for steps in range(1, interp_ratio):
                if sum(sum(mov_img)) > sum(sum(fix_img)):
                    out_slice = (slice) * interp_ratio + steps
                    next_deform = ants.image_read(mytx['invtransforms'][1])
                else:
                    out_slice = (slice + 1) * interp_ratio - steps
                    next_deform = ants.image_read(mytx['fwdtransforms'][0])

                next_deform = next_deform.apply(lambda x: x / interp_ratio * steps)
                next_deform.to_file(temp_filename)

                if sum(sum(mov_img)) > sum(sum(fix_img)):
                    next_mid = ants.apply_transforms(mov_img, fix_img, temp_filename)
                else:
                    next_mid = ants.apply_transforms(fix_img, mov_img, temp_filename)

                next_mid = ants.resample_image(next_mid, (1, 1))
                shapes = nim_roi.label_image_2d_to_polygon_shapes(next_mid.numpy() > 0)
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
    ra_dict2['VoxelSizeXInUM'] = 10
    ra_dict2['VoxelSizeYInUM'] = 10
    ra_dict2['VoxelSizeZInUM'] = 100 / interp_ratio
    region_annotation.write_region_annotation_dict(ra_dict2, result_filename)


def generate_symmetric_mesh(mesh_folder: str):
    # List mesh files
    (_, path_name, file_list) = next(os.walk(mesh_folder))
    file_list = [fn for fn in file_list if '.obj' in fn]

    # Get min-max bound box for all meshes
    bound_box = np.array([[float('inf'), -float('inf')] for i in range(3)])
    for region_name in file_list:
        mesh_filename = os.path.join(mesh_folder, region_name)
        msh = ZMesh(mesh_filename)
        for dim in range(3):
            bound_box[dim][0] = min(bound_box[dim][0], np.min(msh.vertices, axis=0)[dim])
            bound_box[dim][1] = max(bound_box[dim][1], np.max(msh.vertices, axis=0)[dim])

    box_size = bound_box[:,1] - bound_box[:,0]
    box_size[0] *= 2.
    # Flip and relocate meshes based on the bound_box
    for region_name in file_list:
        mesh_filename = os.path.join(mesh_folder, region_name)
        # Open using ZMesh
        logger.info(f'Currently processing region {region_name}')
        msh = ZMesh(mesh_filename)

        mesh_vertices = msh.vertices.copy()
        for dim in range(3):
            mesh_vertices[:,dim] -= bound_box[dim][0]
        msh.vertices = mesh_vertices
        msh.save(mesh_filename)

        flip_mesh_vertices = msh.vertices.copy()
        flip_mesh_vertices[:,0] = -flip_mesh_vertices[:,0]
        # msh.vertices = flip_mesh_vertices
        # mesh_filename = os.path.join(mesh_folder, f'flip_{region_name}')
        # msh.save(mesh_filename)

        msh.vertices = np.concatenate((mesh_vertices, flip_mesh_vertices))
        flip_mesh_normals = msh.normals.copy()
        flip_mesh_normals[:, 0] = -flip_mesh_normals[:, 0]
        msh.normals = np.concatenate((msh.normals, flip_mesh_normals))
        flip_mesh_indices = msh.indices.copy() + np.max(msh.indices) + 1
        msh.indices = np.concatenate((msh.indices, flip_mesh_indices))

        mesh_filename = os.path.join(mesh_folder, f'combined_{region_name}')
        msh.save(mesh_filename)


def convert_mesh_to_isotropic_mask(mesh_folder: str, result_folder: str = None):
    # List mesh files
    (_, path_name, file_list) = next(os.walk(mesh_folder))
    file_list = [fn for fn in file_list if '.obj' in fn and ',' not in fn]

    for region_name in file_list:
        mesh_filename = os.path.join(mesh_folder, region_name)
        result_filename = os.path.join(result_folder, f'{region_name[:-4]}.nim')
        if os.path.exists(result_filename):
            logger.info(f'{region_name} already exist')
        else:
            # Open using ZMesh
            logger.info(f'Currently processing region {region_name}')
            msh = ZMesh(mesh_filename)

            # Rescale and center vertices
            mesh_vertices = msh.vertices
            # mesh_vertices[:, 0] += 14040.
            # mesh_vertices[:, 1] += 672.
            # mesh_vertices[:, 2] += 158.
            # mesh_vertices[:, 0] /= 16
            # mesh_vertices[:, 1] /= 16
            # mesh_vertices[:, 2] /= 16
            mesh_vertices[:, 1] += 200.
            mesh_vertices[:, 2] += 15200.
            mesh_vertices[:, 2] /= 16
            mesh_vertices[:, 0] /= 16
            mesh_vertices[:, 1] /= 16
            msh.vertices = mesh_vertices

            # Voxelize and clear mesh file
            img = msh.toLabelImg(width=1755, height=1268, depth=1800)

            for slice_idx in range(img.data[0][0,:,:,:].shape[0]):
                slice_img = img.data[0][0,slice_idx,:,:].copy()
                if np.sum(slice_img) == 0:
                    continue
                # filled_slice = binary_fill_holes(slice_img).astype(int)
                # holes = opening(filled_slice - slice_img, square(2))
                # filled_slice = filled_slice - holes
                slice_img = opening(slice_img, square(2))
                slice_img = closing(slice_img, square(2))
                # props = measure.regionprops_table(measure.label(slice_img), properties=('area', 'coords'))
                # noise_artfct = [i for i, x in enumerate(props['area']) if x < 300]
                # if len(noise_artfct) > 0:
                #     noise_voxel = np.concatenate(props['coords'][noise_artfct])
                #     noise_voxel = [np.ravel_multi_index(x, slice_img.shape) for x in noise_voxel]
                #     slice_img[np.unravel_index(noise_voxel, slice_img.shape)] = 0
                img.data[0][0,slice_idx,:,:] = slice_img
            img.save(result_filename)


def merge_and_refine_mesh(children_list: list, mask_folder: str, ontology: dict, parent_mask: int = None, expand_px:
    int = 10, refine_px: int = 0):
    if parent_mask is None:
        # Initial masking using parent volume
        parent_zimg = ZImg(os.path.join(mask_folder, children_list[0]))
        merged_mask = parent_zimg.data[0][0, :, :, :].copy()
    else:
        merged_mask = parent_mask
    width = merged_mask.shape[2]
    height = merged_mask.shape[1]
    depth = merged_mask.shape[0]

    region_label = np.zeros((depth, height, width), dtype='uint16') # Fixed size for now
    region_mask = np.zeros((depth, height, width), dtype='uint16') # Fixed size for now
    # Merge and remove overlapping area
    region_list = []
    for region_name in children_list:
        logger.info(f'Loading  {region_name}')
        region_zimg = ZImg(os.path.join(mask_folder, region_name))
        region_id = get_id_from_ontology(ontology, region_name[:-4])
        region_list.append(region_id)
        region_img = region_zimg.data[0][0, :, :, :].copy()

        region_label = region_label + region_img * region_id
        region_mask += region_img
    if parent_mask is None:
        for slice_idx in range(depth):
            logger.info(f'Computing merged mask on slice {slice_idx}')
            slice_img = (region_label[slice_idx, :, :] > 0) * 1.
            slice_img = cv2.morphologyEx(slice_img, cv2.MORPH_CLOSE, disk(5))
            slice_img = binary_fill_holes(slice_img > 0).astype(int)
            merged_mask[slice_idx, :, :] = slice_img

    # Expand children label slightly and refine
    for slice_idx in range(depth):
        logger.info(f'Refining slice {slice_idx}')
        slice_img = region_label[slice_idx, :, :]
        slice_img[region_mask[slice_idx, :, :] > 1] = 0
        slice_img[merged_mask[slice_idx, :, :] == 0] = 0
        slice_img[slice_img == 315] = 0
        slice_img = expand_labels(slice_img, distance=expand_px)
        slice_img[merged_mask[slice_idx, :, :] == 0] = 0
        if(refine_px > 0):
            for region_id in np.unique(slice_img):
                if region_id > 0:
                    temp_mask = (slice_img == region_id) *1.
                    temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, disk(refine_px))
                    slice_img[slice_img == region_id] = temp_mask[slice_img == region_id] * region_id
        slice_img = expand_labels(slice_img, distance=expand_px)
        slice_img[merged_mask[slice_idx, :, :] == 0] = 0
        region_label[slice_idx,:,:] = slice_img
    return region_label


def fix_hierarchical_mesh_structure(mask_folder: str, result_folder: str, ra_filename: str):
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()

    # Fix cortex layer
    (_, path_name, file_list) = next(os.walk(mask_folder))
    children_list = [fn for fn in file_list if 'Layer' in fn]
    parent_region = 'Isocortex.nim'
    refined_layer, merged_layer = merge_and_refine_mesh(children_list, parent_region, mask_folder, ontology,
                                                        expand_px=20, refine_px=0)

    # Save and edit layer annotation
    result_layer_ra = os.path.join(result_folder, '11_final_annotation_for_3d_layer.reganno')
    if not os.path.exists(result_layer_ra):
        mask_to_polygon_annotation(ra_filename, refined_layer, result_layer_ra)
        ra_dict = region_annotation.read_region_annotation(result_layer_ra)
    else:
        ra_dict = region_annotation.read_region_annotation(result_layer_ra)

    # Fix cortex subregion
    (_, path_name, file_list) = next(os.walk(mask_folder))
    children_list = [fn for fn in file_list if 'Area' in fn]
    parent_region = 'Isocortex.nim'
    refined_subregion = merge_and_refine_mesh(children_list, parent_region, mask_folder, ontology,
                                                        expand_px=20, refine_px=10)

    result_layer_ra = os.path.join(result_folder, '11_final_annotation_for_3d_subregion.reganno')
    if not os.path.exists(result_layer_ra):
        mask_to_polygon_annotation(ra_filename, refined_layer, result_layer_ra)
        ra_dict = region_annotation.read_region_annotation(result_layer_ra)
    else:
        ra_dict = region_annotation.read_region_annotation(result_layer_ra)

    # Load annotation and covert back to nim
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    merged_layer = np.zeros((1, 180, 1268, 1755), dtype='uint16')
    merged_cortex = np.zeros((1, 180, 1268, 1755), dtype='uint16')
    for region_id in [3155, 3156, 3157, 3158, 3159]:
        for slice_idx, maskps in region_to_masks[region_id].items():
            mask = np.zeros(shape=(1268, 1755), dtype='uint16')
            for compact_mask, x_start, y_start, _ in maskps:
                mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] = mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] | compact_mask

            merged_slice = merged_layer[0, slice_idx, :, :]
            merged_slice[mask>0] = mask[mask>0] * region_id
            merged_layer[0, slice_idx, :, :] = merged_slice
    merged_cortex = (merged_layer > 0)*1.


    # merged_zimg = ZImg(merged_layer)
    # merged_zimg.save(os.path.join(working_folder, 'Isocortex.nim'))
    # refined_zimg = ZImg(refined_layer)
    # refined_zimg.save(os.path.join(working_folder, 'Layers.nim'))

    # Fix cortex subregion -> save subregion annotation

    # Combine layer + subregion -> save subregion layer annotation
    # Merge all to get cortex -> save cortex annotation


    return


def mask_to_polygon_annotation(ra_filename, refined_layer, result_layer_ra):
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    for region_id, region in ra_dict['Regions'].items():
        region['ROI'] = None
    region_list = np.unique(refined_layer)
    region_list = [idx for idx in region_list if idx != 0]
    for slice_idx in range(refined_layer.shape[1]):
        for region_id in region_list:
            print(f'Refining slice {slice_idx}')
            slice_img = refined_layer[0, slice_idx, :, :]
            region_spline = nim_roi.mask_2d_to_polygon_shapes(slice_img == region_id)
            if ra_dict['Regions'][region_id]['ROI'] is None:
                ra_dict['Regions'][region_id]['ROI'] = {}
            if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()
    region_annotation.write_region_annotation_dict(ra_dict, result_layer_ra)


def fix_reference_2d_subregion_gyrus(ra_filename: str, out_ra_filename: str):
    read_ratio = 4
    scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
    depth = 162
    height = 5072
    width = 7020

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_filename}')
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    logger.info(f'finish reading masks from {ra_filename}')
    ra_dict['Regions'][-1]['ROI'] = None

    cutline_slice_list = list(region_to_masks[-1].keys())

    for slice_idx in cutline_slice_list:
        logger.info(f'fixing slice {slice_idx}')
        parent_subregion_list = [96, 322, 895]
        for layer_idx in range(1,6):
            annotation_mask = np.zeros(shape=(height, width), dtype=np.uint16)
            for parent_idx in parent_subregion_list:
                subregion_idx = parent_idx*10 + layer_idx
                if not slice_idx in region_to_masks[subregion_idx].keys():
                    continue
                ra_dict['Regions'][subregion_idx]['ROI']['SliceROIs'].pop(slice_idx)
                maskps = region_to_masks[subregion_idx][slice_idx]
                for compact_mask, x_start, y_start, _ in maskps:
                    if compact_mask.sum() == 0:
                        continue
                    assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                    mask = np.zeros(shape=(height, width), dtype=np.bool)
                    mask[y_start:y_start + compact_mask.shape[0],
                    x_start:x_start + compact_mask.shape[1]] = compact_mask
                    annotation_mask[mask] = subregion_idx

            for mask_idx in [-1, 315]:
                maskps = region_to_masks[mask_idx][slice_idx]
                mask = np.zeros(shape=(height, width), dtype=np.bool)
                for compact_mask, x_start, y_start, _ in maskps:
                    if compact_mask.sum() == 0:
                        continue
                    assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                    mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = compact_mask
                if mask_idx == 315:
                    annotation_mask[~mask] = 0
                else:
                    annotation_mask[mask] = 0  # cut

            for region_id in np.unique(annotation_mask):
                if region_id == 0:
                    continue
                labeled_array, num_features = scipy.ndimage.label(annotation_mask == region_id)
                for label in range(1, num_features + 1):
                    mask = labeled_array == label
                    # mask = scipy.ndimage.binary_closing(mask, structure=np.ones((5,5)))
                    # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                    #                         kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                    shapes = nim_roi.label_image_2d_to_spline_shapes(mask)
                    if len(shapes) > 0:
                        if ra_dict['Regions'][region_id]['ROI'] is None:
                            ra_dict['Regions'][region_id]['ROI'] = {}
                        if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                        if slice_idx not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = shapes
                        else:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx].extend(shapes)

    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * read_ratio)
    region_annotation.write_region_annotation_dict(ra_dict, out_ra_filename)

    logger.info(f'cut subregions {out_ra_filename} done')


def refine_reference_2d_annotation(ra_filename: str, out_ra_filename: str):
    read_ratio = 4
    scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
    depth = 180
    height = 5072
    width = 7020

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    res_ra_dict = copy.deepcopy(ra_dict)
    logger.info(f'finish reading {ra_filename}')
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    logger.info(f'finish reading masks from {ra_filename}')

    cortex_child_list = search_all_child(ra_dict, [315])

    # Fix cortex-subregion-layer
    cortex_slice_list = list(region_to_masks[315].keys())
    for slice_idx in cortex_slice_list:
    # for slice_idx in range(100,110):
        logger.info(f'fixing layers and cortex subregion structure in slice {slice_idx}')
        cortex_subregion = np.zeros(shape=(height, width), dtype=np.uint16)
        for region_id, slice_rois in region_to_masks.items():
            if region_id not in cortex_child_list or slice_idx not in region_to_masks[region_id].keys():
                continue
            maskps = region_to_masks[region_id][slice_idx]
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                mask = np.zeros(shape=(height, width), dtype=np.bool)
                mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] = compact_mask
                cortex_subregion[mask] = region_id

        # mask using cortex boundary
        maskps = region_to_masks[315][slice_idx]
        cortex_mask = np.zeros(shape=(height, width), dtype=np.bool)

        for compact_mask, x_start, y_start, _ in maskps:
            if compact_mask.sum() == 0:
                continue
            assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
            mask = np.zeros(shape=(height, width), dtype=np.bool)
            mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = compact_mask
            cortex_mask[mask] = 315
        cortex_subregion[~cortex_mask] = 0

        # expand label
        cortex_subregion = expand_labels(cortex_subregion, distance=30)
        cortex_subregion[~cortex_mask] = 0

        regions_in_slice = [id for id in np.unique(cortex_subregion) if id != 0]
        for region_id in regions_in_slice:
            logger.info(f'Refining region {region_id} in slice {slice_idx}')
            labeled_array, num_features = scipy.ndimage.label(cortex_subregion == region_id)
            res_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx)
            for label in range(1, num_features + 1):
                mask = labeled_array == label
                shapes = nim_roi.label_image_2d_to_polygon_shapes(mask)
                if len(shapes) > 0:
                    if slice_idx not in res_ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                        res_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = shapes
                    else:
                        res_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx].extend(shapes)

    # Fix all others
    ignore_region_list = cortex_child_list + [315]
    for slice_idx in range(depth):
    # for slice_idx in range(105, 110):
        annotation_mask = np.zeros(shape=(height, width), dtype=np.uint16)
        count_mask = np.zeros(shape=(height, width), dtype=np.uint16)
        for region_id, slice_rois in region_to_masks.items():
            if region_id in ignore_region_list or slice_idx not in region_to_masks[region_id].keys():
                continue
            maskps = slice_rois[slice_idx]
            logger.info(f'Processing region {region_id} in slice {slice_idx}')
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                mask = np.zeros(shape=(height, width), dtype=np.bool)
                mask[y_start:y_start + compact_mask.shape[0],
                x_start:x_start + compact_mask.shape[1]] = compact_mask
                mask = cv2.morphologyEx((mask)*1., cv2.MORPH_OPEN, disk(3))>0
                annotation_mask[mask] = region_id
                count_mask[mask] += 1

        if slice_idx in region_to_masks[315].keys():
            maskps = region_to_masks[315][slice_idx]
            cortex_mask = np.zeros(shape=(height, width), dtype=np.bool)
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                cortex_mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = compact_mask
            annotation_mask[cortex_mask>0] = 315.

        middle_mask = np.zeros(annotation_mask.shape, dtype='bool')
        middle_mask[:, round(width/2)-70:round(width/2)+70] = annotation_mask[:, round(width/2)-70:round(width/2)+70]\
                                                              == 0
        region_mask = cv2.morphologyEx((annotation_mask > 0)*1., cv2.MORPH_CLOSE, disk(15))
        region_mask = binary_fill_holes(region_mask > 0)
        if slice_idx < 35:
            region_mask[middle_mask] = 0
        annotation_mask[count_mask > 1] = 0
        annotation_mask[cortex_mask > 0] = 0
        regions_in_slice = [id for id in np.unique(annotation_mask) if id != 0 and id != 315]
        for region_id in regions_in_slice:
            if region_id in [9101, 9102, 9103]:
                continue
            mask = annotation_mask == region_id
            mask = cv2.morphologyEx(mask * region_id, cv2.MORPH_OPEN, disk(5))
            annotation_mask[annotation_mask == region_id] = mask[annotation_mask == region_id]
        annotation_mask = expand_labels(annotation_mask, distance=30)
        annotation_mask[~region_mask] = 0

        regions_in_slice = [id for id in np.unique(annotation_mask) if id != 0 and id != 315]
        for region_id in regions_in_slice:
            if region_id in [9101, 9102, 9103]:
                continue
            mask = annotation_mask == region_id
            mask = cv2.morphologyEx(mask*region_id, cv2.MORPH_OPEN, disk(10))
            annotation_mask[annotation_mask == region_id] = mask[annotation_mask == region_id]
        annotation_mask = expand_labels(annotation_mask, distance=20)
        annotation_mask[~region_mask] = 0
        annotation_mask[cortex_mask > 0] = 0

        for region_id in regions_in_slice:
            logger.info(f'Refining region {region_id} in slice {slice_idx}')
            labeled_array, num_features = scipy.ndimage.label(annotation_mask == region_id)
            res_ra_dict['Regions'][region_id]['ROI']['SliceROIs'].pop(slice_idx)
            for label in range(1, num_features + 1):
                mask = labeled_array == label
                shapes = nim_roi.label_image_2d_to_polygon_shapes(mask)
                if len(shapes) > 0:
                    if slice_idx not in res_ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                        res_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx] = shapes
                    else:
                        res_ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice_idx].extend(shapes)

    res_ra_dict = region_annotation.transform_region_annotation_dict(res_ra_dict, lambda coords: coords * read_ratio)
    region_annotation.write_region_annotation_dict(res_ra_dict, out_ra_filename)


def generate_hierarchical_mesh_from_mask(mask_folder:str, result_folder: str = None):
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()

    if result_folder is None:
        result_folder = os.path.join(mask_folder, '../../mesh')

    # Load all area mask
    (_, path_name, file_list) = next(os.walk(mask_folder))
    file_list = [fn for fn in file_list if '.nim' in fn]
    id_list = [get_id_from_ontology(ontology, fn[:-4]) for fn in file_list]
    file_id_list = list(zip(id_list, file_list))

    def generate_mesh_from_mask(file_id_list:list, current_list:list, st_level:int, mask_folder:str, result_folder:str):
        id_list, file_list = zip(*file_id_list)
        id_list = list(id_list)
        file_list = list(file_list)
        if len(current_list) == 0:
            return
        region_list = [x['id'] for x in current_list]
        for region_id in region_list:
            children_list = [x['children'] for x in current_list if x['id'] == region_id][0]
            if region_id in id_list:
                filename = file_list[id_list.index(region_id)]
                result_filename = os.path.join(result_folder, filename[:-4] + '.obj')
                if os.path.exists(result_filename):
                    logger.info(f'{filename} already exist!')
                    continue
                else:
                    logger.info(f'Generating mesh for {filename} at level {st_level}')
                region_zimg = ZImg(os.path.join(mask_folder, filename))
                region_img = region_zimg.data[0][0, 30:, :, :].copy()
                if np.sum(region_img) == 0:
                    logger.info(f'{filename} is empty')
                    continue
                verts, faces, normals, values = measure.marching_cubes(region_img, step_size=3)
                verts += normals * (2-st_level) * 1
                points = np.array(verts)[:, [2, 1, 0]]
                cells = [("triangle", np.array(faces))]
                meshio.write_points_cells(result_filename, points, cells)
                generate_mesh_from_mask(file_id_list, children_list, st_level+1, mask_folder, result_folder)
            else:
                generate_mesh_from_mask(file_id_list, children_list, st_level, mask_folder, result_folder)

    generate_mesh_from_mask(file_id_list, ontology['children'], 0, mask_folder, result_folder)


if __name__ == "__main__":
    lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')
    hyungju_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align'
    reference_folder = os.path.join(hyungju_folder, 'all-dataset', 'Hotsauce_SMI99_VGluT2_NeuN')
    mesh_folder = os.path.join(reference_folder, 'Mesh_final', 'Processed')
    working_folder = os.path.join(hyungju_folder, 'reference-data', 'mesh-high-res')
    mask_folder = os.path.join(working_folder, 'mesh-mask')

    # region Voxelize mesh to mask
    # convert_mesh_to_mask(mesh_folder, '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/reference-data/mesh-mask')
    # endregion

    # region Refine layer structures
    ra_filename = os.path.join(working_folder, '11_final_annotation_for_3d.reganno')
    result_filename = os.path.join(working_folder, '11_final_annotation_for_3d_layer.reganno')
    # refine_3d_isotropic_cortex_layer(mask_folder, result_filename=result_filename, ra_filename=ra_filename)
    # endregion

    # region Refine cortical subregion
    ra_filename = os.path.join(working_folder, '11_final_annotation_for_3d.reganno')
    result_filename = os.path.join(working_folder,'11_final_annotation_for_3d_subregion.reganno')
    # refine_3d_cortex_subregion(mask_folder, result_filename=result_filename, ra_filename=ra_filename)
    # refine_3d_isotropic_cortex_subregion(mask_folder, result_filename=result_filename, ra_filename=ra_filename)
    # endregion

    # region Merge refined layers and subregions
    # layer_ra_filename = os.path.join(working_folder, '11_final_layer.reganno')
    # subregion_ra_filename = os.path.join(working_folder, '11_final_subregion.reganno')
    ra_filename = os.path.join(working_folder, '11_final_annotation_for_3d.reganno')
    result_filename = os.path.join(working_folder, '11_final_annotation_for_3d_layer_subregion.reganno')
    resolved_folder = os.path.join(mask_folder, 'resolved')
    # merge_subregion_layer(layer_ra_filename=layer_ra_filename, subregion_ra_filename=subregion_ra_filename,
    #                       result_filename=result_filename)
    # merge_isotropic_subregion_layer(mask_folder=resolved_folder, ra_filename=ra_filename, result_filename=result_filename)
    # endregion

    # region Merge refined layers and subregions
    # layer_ra_filename = os.path.join(working_folder, '11_final_layer.reganno')
    # subregion_ra_filename = os.path.join(working_folder, '11_final_subregion.reganno')
    ra_filename = os.path.join(working_folder, '11_final_annotation_for_3d.reganno')
    result_filename = os.path.join(working_folder, '11_final_annotation_for_3d_subcortical.reganno')
    # resolved_folder = os.path.join(mask_folder, 'resolved')
    mask_folder = os.path.join(working_folder, 'mesh-mask')
    # refine_3d_isotropic_all_subregion(mask_folder, result_filename, ra_filename)
    # endregion

    ra_filename = os.path.join(working_folder, '11_final_annotation_for_3d_final.reganno')
    result_filename = os.path.join(working_folder, '99_final_annotation_for_3d.reganno')
    # refine_3d_annotation_final(ra_filename, result_filename,resolved_folder)


    result_folder = os.path.join(resolved_folder, 'mesh')
    generate_hierarchical_mesh_from_mask(resolved_folder, result_folder)

    # region Interp 3D annotation
    # ra_filename = os.path.join(working_folder, '11_final_layer_subregion.reganno')
    # result_filename = os.path.join(working_folder, '11_final_layer_subregion_interp.reganno')
    # if sys.platform == 'linux':
    #     folder = '/home/hyungjujeon/hyungju/data'
    #     ra_filename = os.path.join(folder, 'mesh-interp','11_final_layer_subregion.reganno')
    #     result_filename = os.path.join(folder, 'mesh-interp', '11_final_layer_subregion_interp.reganno')
    #
    # interp_3d_layer(ra_filename, result_filename)
    #endregion

    # region generate symmetric mesh
    #mirror_mesh_folder = os.path.join(working_folder, 'mesh', 'processed', 'mirror')
    # generate_symmetric_mesh(mirror_mesh_folder)
    # endregion

    # Below this line uses eLemur v1.2
    mesh_folder = os.path.join('/Volumes/shared_2/Project/Mouse_Lemur/8_processed_results/0_final_mesh/ver-1_1')
    working_folder = os.path.join(hyungju_folder, 'reference-data', 'mesh-high-res')
    mask_folder = os.path.join(working_folder, 'mesh-mask')
    if sys.platform == 'linux':
        mesh_folder = os.path.join(io.jinny_nas_dir(), 'Mouse_Lemur', '8_processed_results', '0_final_mesh', 'ver-1_1')
        folder = '/home/hyungjujeon/hyungju/data'
        ra_filename = os.path.join(folder, 'mesh_processing', '11_final_3d_annotation.reganno')
        result_filename = os.path.join(folder, 'mesh_processing', '11_final_3d_annotation_layer.reganno')
        mask_folder = os.path.join(folder, 'mesh_processing', 'mesh')

    # region Voxelize mesh to isotropic mask
    # convert_mesh_to_isotropic_mask(mesh_folder, mask_folder)
    # endregion

    # region refine_2d_annotation
    ra_filename = os.path.join(reference_folder, '00_stacked_annotation_layer_fix.reganno')
    result_filename = os.path.join(reference_folder, '00_stacked_annotation_layer_fix_resolve.reganno')
    if sys.platform == 'linux':
        folder = '/home/hyungjujeon/hyungju/data'
        ra_filename = os.path.join(folder, 'mesh_processing', '00_stacked_annotation_layer_fix.reganno')
        result_filename = os.path.join(folder, 'mesh_processing', '00_stacked_annotation_layer_fix_resolve.reganno')

    refine_reference_2d_annotation(ra_filename, result_filename)
    # endregion

    # refine_3d_cortex
    ra_filename = os.path.join(working_folder, '11_final_annotation_for_3d.reganno')
    # fix_hierarchical_mesh_structure(mask_folder, working_folder, ra_filename)
    # verts, faces, normals, values = measure.marching_cubes(temp_img, step_size=2)
    # points = np.array(verts)[:,[2,1,0]]
    # cells = [("triangle", np.array(faces))]
    # meshio.write_points_cells('/home/hyungjujeon/hyungju/data/Caudate Nucleus_2.obj',points,cells)
