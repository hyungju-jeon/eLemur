import os
import sys
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
import trimesh
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa


from zimg import *
from utils import io
from utils import img_util
from utils import nim_roi
from utils import region_annotation
from utils.logger import setup_logger


logger = setup_logger()

# def calibrate_mesh():
#     return calibrate_mesh

    
def get_id(tree_root: dict, name: str):
    if len(tree_root['children']) > 0:
        for child_tree in tree_root['children']:
            if child_tree['name'] == name:
                return child_tree['id'] 
            else:
                region_id = get_id(child_tree,name)
            if region_id is not None:
                return region_id

def run_mesh_to_reganno(ra_filename:str, mesh_folder:str):
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()
    
    # Get region annotation
    scale_down = 1.0 / 16  # otherwise the mask will be too big
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)

    #%% Get all mesh list 
    unprocessed_mesh_list = glob.glob(os.path.join(mesh_folder, "*.obj"))
    region_mask = np.zeros(shape=(180,1268,1755), dtype=np.uint8)
    for mesh_filename in unprocessed_mesh_list:
        # Get unprocessed mesh information
        mesh_name = os.path.basename(mesh_filename)[:-4]
        processed_mesh_filename = os.path.join(processed_folder, f'{mesh_name}.obj')
        
        if not os.path.isfile(processed_mesh_filename):
            continue
            
        logger.info(f'Currently running region {mesh_name}')
        mesh_id = get_id(ontology, mesh_name)
        
        msh = trimesh.load(mesh_filename)
        mesh_bound = (min(msh.vertices[:,0]), min(msh.vertices[:,1]), min(msh.vertices[:,2]))
        
        # Update and reorient processed mesh
        msh = trimesh.load(processed_mesh_filename)
        
        # Rescale vertices
        msh.vertices[:,2] /= 160
        msh.vertices[:,2] += 92.

        msh.vertices[:,0] /= 16
        msh.vertices[:,1] /= 16
        
        temp_filename = os.path.join(processed_folder, 'temp.obj')
        msh.export(temp_filename)
        
        # Open using ZMesh and voxelize
        msh = ZMesh(temp_filename)
        img = msh.toLabelImg(width = 1755, height = 1268, depth = 180)
        
        ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'].clear()
        # Get voxel Image
        for slice_idx in range(0,180):
            if sum(sum(img.data[0][0,slice_idx,:,:])) == 0:
                continue
            logger.info(f'Currently running region {mesh_name}, slice{slice_idx}')
            slice_image = ants.from_numpy(np.array(img.data[0][0,slice_idx,:,:]))
            slice_image = ants.get_mask(slice_image,cleanup = 0)
            
            # Morphologial operations to remove noise
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='open', radius=3, shape='ball')
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='close', radius=4, shape='ball')
            region_mask[slice_idx,:,:] += slice_image.numpy().astype('uint8')
            # slice_image = ants.iMath(slice_image, 'FillHoles')
            slice_image = slice_image.numpy()
            
            region_spline = nim_roi.mask_2d_to_spline_shapes(slice_image > 0)
            ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()
            
    scale_down = 16.  # otherwise the mask will be too big
    ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)       
    region_annotation.write_region_annotation_dict(ra_dict2, f'/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN/07_meshmixer_processed_layer.reganno')   
    
def run_mesh_to_expanded_reganno(ra_filename:str, mesh_folder:str):
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()
    
    # Get region annotation
    scale_down = 1.0 / 16  # otherwise the mask will be too big
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)

    #%% Get all mesh list 
    unprocessed_mesh_list = glob.glob(os.path.join(mesh_folder, "*.obj"))
    region_mask = np.zeros(shape=(180,1268,1755), dtype=np.uint8)
    for mesh_filename in unprocessed_mesh_list:
        # Get unprocessed mesh information
        mesh_name = os.path.basename(mesh_filename)[:-4]
        processed_mesh_filename = os.path.join(processed_folder, f'{mesh_name}.obj')
        
        if not os.path.isfile(processed_mesh_filename):
            continue
            
        logger.info(f'Currently running region {mesh_name}')
        mesh_id = get_id(ontology, mesh_name)
        
        msh = trimesh.load(mesh_filename)
        mesh_bound = (min(msh.vertices[:,0]), min(msh.vertices[:,1]), min(msh.vertices[:,2]))
        
        # Update and reorient processed mesh
        msh = trimesh.load(processed_mesh_filename)
        
        # Rescale vertices
        msh.vertices[:,2] /= 160
        msh.vertices[:,2] += 93.5

        msh.vertices[:,0] /= 16
        msh.vertices[:,1] /= 16
        
        temp_filename = os.path.join(processed_folder, 'temp.obj')
        msh.export(temp_filename)
        
        # Open using ZMesh and voxelize
        msh = ZMesh(temp_filename)
        img = msh.toLabelImg(width = 1755, height = 1268, depth = 180)
        
        ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'].clear()
        # Get voxel Image
        for slice_idx in range(0,180):
            if sum(sum(img.data[0][0,slice_idx,:,:])) == 0:
                continue
            logger.info(f'Currently running region {mesh_name}, slice{slice_idx}')
            slice_image = ants.from_numpy(np.array(img.data[0][0,slice_idx,:,:]))
            slice_image = ants.get_mask(slice_image,cleanup = 0)
            
            # Morphologial operations to remove noise
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='open', radius=2, shape='ball')
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='close', radius=4, shape='ball')
            region_mask[slice_idx,:,:] += slice_image.numpy().astype('uint8')
            # slice_image = ants.iMath(slice_image, 'FillHoles')
            slice_image = slice_image.numpy()
            
            region_spline = nim_roi.mask_2d_to_spline_shapes(slice_image > 0)
            ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()
    
        
    #%% Get fibertract mesh without expanding
    region_fiber_mask = region_mask.copy()       
    fiber_meshname = os.path.join(mesh_folder, 'fiber tracts.obj')
    msh = trimesh.load(fiber_meshname)
    mesh_id = get_id(ontology, 'fiber tracts')
            
    msh.vertices[:,0] /= 16
    msh.vertices[:,1] /= 16
    
    temp_filename = os.path.join(processed_folder, 'temp.obj')
    msh.export(temp_filename)
    
    msh = ZMesh(temp_filename)
    img = msh.toLabelImg(width = 1755, height = 1268, depth = 180)
    
    # Run processing       
    ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'].clear()
    for slice_idx in range(0,180):
        if sum(sum(img.data[0][0,slice_idx,:,:])) == 0:
            continue
        display(f'Currently callibrating fiber tracts, slice{slice_idx}')
        fiber_image = ants.from_numpy(np.array(img.data[0][0,slice_idx,:,:]))
        fiber_image = ants.get_mask(fiber_image,cleanup = 0) 
        
        # Morphologial operations to remove noise
        fiber_image = ants.morphology(fiber_image, mtype='binary', value=1, operation='open', radius=3, shape='ball')
        fiber_image = ants.morphology(fiber_image, mtype='binary', value=1, operation='dilate', radius=3, shape='ball')
        fiber_image = ants.morphology(fiber_image, mtype='binary', value=1, operation='close', radius=3, shape='ball')
        region_fiber_mask[slice_idx,:,:] += fiber_image.numpy().astype('uint8')        
        fiber_image = fiber_image.numpy()
        
    scale_down = 16.  # otherwise the mask will be too big     
    ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)       
    region_annotation.write_region_annotation_dict(ra_dict2, f'/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN/05_meshmixer_processed_annotation_2.reganno')  

    #%% Slightly Expand inner regions     
    region_expand_mask = region_mask.copy()
    region_expand_mask = region_expand_mask.astype('uint16')
    expand_region_list = [1,2,3,9,12,15,16,17,19]
    for expand_region in expand_region_list:
        mesh_filename = unprocessed_mesh_list[expand_region]
        mesh_name = os.path.basename(mesh_filename)[:-4]
        processed_mesh_filename = os.path.join(processed_folder, f'{mesh_name}.obj')
        
        if not os.path.isfile(processed_mesh_filename):
            continue
            
        logger.info(f'Currently running region {mesh_name}')
        mesh_id = get_id(ontology, mesh_name)
        
        msh = trimesh.load(mesh_filename)
        mesh_bound = (min(msh.vertices[:,0]), min(msh.vertices[:,1]), min(msh.vertices[:,2]))
        
        # Update and reorient processed mesh
        msh = trimesh.load(processed_mesh_filename)
        
        # Rescale vertices
        msh.vertices[:,2] /= 160
        msh.vertices[:,2] += 93.5
            
        msh.vertices[:,0] /= 16
        msh.vertices[:,1] /= 16
        
        temp_filename = os.path.join(processed_folder, 'temp.obj')
        msh.export(temp_filename)
        
        # Open using ZMesh and voxelize
        msh = ZMesh(temp_filename)
        img = msh.toLabelImg(width = 1755, height = 1268, depth = 180)
        
        # Run processing        
        ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'].clear()
        for slice_idx in range(0,180):
            if sum(sum(img.data[0][0,slice_idx,:,:])) == 0:
                continue
            display(f'Currently callibrating region {mesh_name}, slice{slice_idx}')
            slice_image = ants.from_numpy(np.array(img.data[0][0,slice_idx,:,:]))
            slice_image = ants.get_mask(slice_image,cleanup = 0) 
            
            # Morphologial operations to remove noise
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='open', radius=2, shape='ball')
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='close', radius=4, shape='ball')
            expand_image = ants.morphology(slice_image, mtype='binary', value=1, operation='dilate', radius=4, shape='ball')
            
            region_image = ants.from_numpy(region_fiber_mask[slice_idx,:,:])
            region_image = ants.get_mask(region_image,cleanup = 0)
            brain_mask = ants.morphology(region_image, mtype='binary', value=1, operation='dilate', radius=15, shape='ball') 
            brain_mask = brain_mask.numpy()
            brain_mask[:,1:877] = 0
            region_image = region_image.numpy()
            
            expand_image = expand_image.numpy()
            slice_image = slice_image.numpy()
            slice_image = np.logical_or(slice_image>0,
                                        np.logical_and(brain_mask>0, 
                                                       np.logical_and(expand_image>0, 
                                                                      np.logical_not(np.logical_or(region_image>0, 
                                                                                                   region_expand_mask[slice_idx,:,:]>0)
                                                                                     )
                                                                      )
                                                       )
                                        )
            slice_image = ants.morphology(ants.from_numpy(slice_image.astype('uint8')), mtype='binary', value=1, operation='open', radius=2, shape='ball')
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='close', radius=4, shape='ball')
            
            slice_image = slice_image.numpy().astype('uint8')
            temp = region_expand_mask[slice_idx,:,:]
            temp[slice_image>0] = mesh_id
            region_expand_mask[slice_idx,:,:] = temp.astype('uint16')
            if sum(sum(slice_image>0)) == 0:
                continue
            
            
            region_spline = nim_roi.mask_2d_to_spline_shapes(slice_image)
            ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()   
    
    scale_down = 16.  # otherwise the mask will be too big
    ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)       
    region_annotation.write_region_annotation_dict(ra_dict2, f'/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN/05_meshmixer_processed_annotation_3.reganno')       
    
    #%% Greatly Expand inner regions     
    region_expand_second_mask = region_expand_mask.copy()
    expand_region_list = [1,2,3,9,12,15,16,17,19]
    for expand_region in expand_region_list:
        mesh_filename = unprocessed_mesh_list[expand_region]
        mesh_name = os.path.basename(mesh_filename)[:-4]
        processed_mesh_filename = os.path.join(processed_folder, f'{mesh_name}.obj')
        
        if not os.path.isfile(processed_mesh_filename):
            continue
            
        logger.info(f'Currently running region {mesh_name}')
        mesh_id = get_id(ontology, mesh_name)
        
        msh = trimesh.load(mesh_filename)
        mesh_bound = (min(msh.vertices[:,0]), min(msh.vertices[:,1]), min(msh.vertices[:,2]))
        
        # Update and reorient processed mesh
        msh = trimesh.load(processed_mesh_filename)
        
        # Rescale vertices
        msh.vertices[:,2] /= 160
        msh.vertices[:,2] += 93.5
            
        msh.vertices[:,0] /= 16
        msh.vertices[:,1] /= 16
        
        temp_filename = os.path.join(processed_folder, 'temp.obj')
        msh.export(temp_filename)
        
        # Open using ZMesh and voxelize
        msh = ZMesh(temp_filename)
        img = msh.toLabelImg(width = 1755, height = 1268, depth = 180)
        
        # Run processing        
        ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'].clear()
        for slice_idx in range(0,180):
            if sum(sum(img.data[0][0,slice_idx,:,:])) == 0:
                continue
            display(f'Currently callibrating region {mesh_name}, slice{slice_idx}')
            slice_image = ants.from_numpy(np.array(img.data[0][0,slice_idx,:,:]))
            slice_image = ants.get_mask(slice_image,cleanup = 0) 
            
            # Morphologial operations to remove noise
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='open', radius=2, shape='ball')
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='close', radius=4, shape='ball')
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='dilate', radius=4, shape='ball')
            expand_image = ants.morphology(slice_image, mtype='binary', value=1, operation='dilate', radius=6, shape='ball')
            
            region_image = ants.from_numpy(region_fiber_mask[slice_idx,:,:])
            region_image = ants.get_mask(region_image,cleanup = 0)
            brain_mask = ants.morphology(region_image, mtype='binary', value=1, operation='dilate', radius=5, shape='ball') 
            brain_mask = ants.morphology(region_image, mtype='binary', value=1, operation='close', radius=30, shape='ball') 
            brain_mask = ants.get_mask(brain_mask)
            brain_mask = brain_mask.numpy()
            brain_mask[:,1:877] = 0
            region_image = region_image.numpy()
            
            expand_image = expand_image.numpy()
            slice_image = slice_image.numpy()
            
            slice_image = np.logical_or(region_expand_mask[slice_idx,:,:]==mesh_id,
                                        np.logical_and(brain_mask>0, 
                                                       np.logical_and(expand_image>0, 
                                                                      np.logical_not(np.logical_or(region_image>0, 
                                                                                                   region_expand_second_mask[slice_idx,:,:]>0)
                                                                                     )
                                                                      )
                                                       )
                                        )
            slice_image = ants.morphology(ants.from_numpy(slice_image.astype('uint8')), mtype='binary', value=1, operation='open', radius=2, shape='ball')
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='close', radius=4, shape='ball')
            
            slice_image = slice_image.numpy().astype('uint8')  
            region_expand_second_mask[slice_idx,:,:] += slice_image
            if sum(sum(slice_image>0)) == 0:
                continue
            
            
            region_spline = nim_roi.mask_2d_to_spline_shapes(slice_image)
            ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()    
    
        # scale_down = 16.  # otherwise the mask will be too big
        # ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)       
        # region_annotation.write_region_annotation_dict(ra_dict2, f'/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN/05_meshmixer_processed_annotation_{mesh_id}.reganno')      
    scale_down = 16.  # otherwise the mask will be too big
    ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)       
    region_annotation.write_region_annotation_dict(ra_dict2, f'/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN/05_meshmixer_processed_annotation_4.reganno')      
    
    #%% Get fibertract mesh with expanding       
    fiber_meshname = os.path.join(mesh_folder, 'fiber tracts.obj')
    msh = trimesh.load(fiber_meshname)
    mesh_id = get_id(ontology, 'fiber tracts')
            
    msh.vertices[:,0] /= 16
    msh.vertices[:,1] /= 16
    
    temp_filename = os.path.join(processed_folder, 'temp.obj')
    msh.export(temp_filename)
    
    msh = ZMesh(temp_filename)
    img = msh.toLabelImg(width = 1755, height = 1268, depth = 180)
    
    #Run processing        
    ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'].clear()
    for slice_idx in range(0,180):
        if sum(sum(img.data[0][0,slice_idx,:,:])) == 0:
            continue
        logger.info(f'Currently callibrating fiber tracts, slice{slice_idx}')
        fiber_image = ants.from_numpy(np.array(img.data[0][0,slice_idx,:,:]))
        fiber_image = ants.get_mask(fiber_image, cleanup = 0) 
        
        # Morphologial operations to remove noise
        fiber_image = ants.morphology(fiber_image, mtype='binary', value=1, operation='open', radius=3, shape='ball')
        fiber_image = ants.morphology(fiber_image, mtype='binary', value=1, operation='close', radius=3, shape='ball')
        
        region_image = ants.from_numpy(region_expand_second_mask[slice_idx,:,:].astype('uint8'))
        region_image = ants.get_mask(region_image>0, cleanup = 0)
        brain_mask = ants.morphology(region_image+fiber_image, mtype='binary', value=1, operation='close', radius=50, shape='ball') 
        brain_mask = ants.morphology(region_image+fiber_image, mtype='binary', value=1, operation='dilate', radius=3, shape='ball')
        brain_mask = ants.get_mask(brain_mask)
        brain_mask = brain_mask.numpy()
        brain_mask[:,1:877] = 0
        region_image = region_image.numpy()
        
        fiber_image = ants.morphology(fiber_image, mtype='binary', value=1, operation='dilate', radius=25, shape='ball')
        fiber_image = fiber_image.numpy()
        fiber_image = np.logical_and(np.logical_and(fiber_image>0, np.logical_not(region_image)),brain_mask>0)
        
        fiber_image = ants.morphology(ants.from_numpy(fiber_image.astype('uint8')), mtype='binary', value=1, operation='open', radius=3, shape='ball')
        fiber_image = ants.morphology(fiber_image, mtype='binary', value=1, operation='close', radius=4, shape='ball')
        
        fiber_image = fiber_image.numpy().astype('uint8')  
        
        if sum(sum(fiber_image>0)) == 0:
            continue
        
        region_spline = nim_roi.mask_2d_to_spline_shapes(fiber_image)
        ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()    
        
    
    scale_down = 16.  # otherwise the mask will be too big
    ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)       
    region_annotation.write_region_annotation_dict(ra_dict2, '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN/05_meshmixer_processed_annotation_final.reganno')    
                
def process_layers(ra_filename:str, mesh_folder:str):
    # Get region annotation
    scale_down = 1.0 / 16  # otherwise the mask will be too big
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)

    #%% Get layer mesh list 
    unprocessed_mesh_list = glob.glob(os.path.join(mesh_folder, "*.obj"))
    region_mask = np.zeros(shape=(180,1268,1755), dtype=np.uint8)
    for mesh_filename in unprocessed_mesh_list:
        # Get unprocessed mesh information
        mesh_name = os.path.basename(mesh_filename)[:-4]
        processed_mesh_filename = os.path.join(processed_folder, f'{mesh_name}.obj')
        
        if not os.path.isfile(processed_mesh_filename):
            continue
            
        logger.info(f'Currently running region {mesh_name}')
        mesh_id = get_id(ontology, mesh_name)
        
        msh = trimesh.load(mesh_filename)
        mesh_bound = (min(msh.vertices[:,0]), min(msh.vertices[:,1]), min(msh.vertices[:,2]))
        
        # Update and reorient processed mesh
        msh = trimesh.load(processed_mesh_filename)
        
        # Rescale vertices
        msh.vertices[:,2] /= 160
        msh.vertices[:,2] += 93.5

        msh.vertices[:,0] /= 16
        msh.vertices[:,1] /= 16
        
        temp_filename = os.path.join(processed_folder, 'temp.obj')
        msh.export(temp_filename)
        
        # Open using ZMesh and voxelize
        msh = ZMesh(temp_filename)
        img = msh.toLabelImg(width = 1755, height = 1268, depth = 180)
        
        ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'].clear()
        # Get voxel Image
        for slice_idx in range(0,180):
            if sum(sum(img.data[0][0,slice_idx,:,:])) == 0:
                continue
            logger.info(f'Currently running region {mesh_name}, slice{slice_idx}')
            slice_image = ants.from_numpy(np.array(img.data[0][0,slice_idx,:,:]))
            slice_image = ants.get_mask(slice_image,cleanup = 0)
            
            # Morphologial operations to remove noise
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='open', radius=2, shape='ball')
            slice_image = ants.morphology(slice_image, mtype='binary', value=1, operation='close', radius=4, shape='ball')
            region_mask[slice_idx,:,:] += slice_image.numpy().astype('uint8')
            # slice_image = ants.iMath(slice_image, 'FillHoles')
            slice_image = slice_image.numpy()
            
            region_spline = nim_roi.mask_2d_to_spline_shapes(slice_image > 0)
            ra_dict['Regions'][mesh_id]['ROI']['SliceROIs'][slice_idx] = region_spline.copy()
    

if __name__ == "__main__":    
    ra_filename = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN/04_scaled_deformed_annotation.reganno'
    mesh_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN/Mesh'
    processed_folder = os.path.join(mesh_folder, 'Processed')
    
    # run_mesh_to_expanded_reganno(ra_filename, mesh_folder)
    
    ra_filename = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN/05_meshmixer_processed_annotation_final.reganno'
    mesh_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/all-dataset/Hotsauce_SMI99_VGluT2_NeuN/Mesh_ver2'
    processed_folder = os.path.join(mesh_folder, 'Processed')
    
    run_mesh_to_reganno(ra_filename, mesh_folder)
            
        
