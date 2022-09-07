import json

import numpy as np
import scipy.io
import scipy.ndimage
import pandas as pd
import re
import ants

from zimg import *
from utils import io
from utils import img_util
from utils import region_annotation
from utils.logger import setup_logger
from utils.brain_info import read_brain_info
from utils.lemur_ontology import *
from skimage import measure
from itertools import groupby

logger = setup_logger()


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def get_flip_id(group_id: int, region_list: list):
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


def convert_key_to_int(str_dict: dict, ):
    int_dict = {}
    for slice_idx, cells in str_dict.items():
        for ch, cell in cells.items():
            if slice_idx not in int_dict.keys():
                int_dict[int(slice_idx)] = {}
            if ch not in int_dict[int(slice_idx)].keys():
                int_dict[int(slice_idx)][int(ch)] = cell.tolist()
    return int_dict


def get_ontology():
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()

    return ontology


def find_valid_cells(slice_cells: np.ndarray, mask_slice: np.array, scale_down: int):
    detected_cells = (slice_cells >= 0).all(1)
    in_slice_cells = np.logical_and(slice_cells[:, 1] * scale_down <= mask_slice.shape[0],
                                    slice_cells[:, 0] * scale_down <= mask_slice.shape[1])
    valid_cells = detected_cells * in_slice_cells
    return valid_cells


def extract_slice_from_region_mask(region_to_masks: dict, target_slice: int, height: int, width: int):
    height = int(height)
    width = int(width)
    annotation_mask = np.zeros(shape=(height, width), dtype=np.uint16)
    for region_id, slice_rois in region_to_masks.items():
        if target_slice in slice_rois.keys():
            maskps = slice_rois[target_slice]
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                mask = np.zeros(shape=(height, width), dtype=np.bool)
                # print(mask.shape, y_start, x_start, compact_mask.shape)
                mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = \
                    compact_mask[0:min(mask.shape[0], y_start + compact_mask.shape[0]) - y_start,
                    0:min(mask.shape[1], x_start + compact_mask.shape[1]) - x_start]

                annotation_mask[mask] = region_id
        else:
            continue
    return annotation_mask


def save_cell_to_swc(cell_centroids: int, save_filename: str):
    with open(save_filename, 'w') as f:
        for i in range(0, len(cell_centroids), 10):
            # cent = slice_cells[i]
            cent = cell_centroids[i]
            f.write(f'{cent[0]} {cent[1]}\n')


def register_cells(folder: str, tform_folder: str, result_folder: str, group_flip: str,
                   is_reference: bool = False, brain_info: dict = None):
    result_filename = os.path.join(result_folder, '00_cell.json')
    if os.path.exists(result_filename):
        logger.info(f'Cell mapping result exists')
        return

    (_, _, filenames) = next(os.walk(os.path.join(folder)))
    r = re.compile('.*([0-9]{2})_s([0-9]+)_ch([0-9]+)_(detection.nim)$')
    filenames = list(filter(r.match, filenames))
    prefix = re.split('(.*)_([0-9]{2})_s([0-9]+)_ch([0-9]+)_(detection.nim)$', filenames[0])[1]
    ch_list = [int(re.split('(.*)_([0-9]{2})_s([0-9]+)_ch([0-9]+)_(detection.nim)$', fn)[-3]) for fn in filenames]
    ch_list = np.unique(ch_list)

    if brain_info is not None:
        nim_filenames = [os.path.split(x)[1] for x in brain_info['filename']]
        scene_list = [int(x) - 1 for x in brain_info['scene']]
        fid_list = [int(re.split('.*_([0-9]+).czi$', fn)[1]) for fn in nim_filenames]
    else:
        fid_list = [int(re.split('(.*)_([0-9]{2})_s([0-9]+)_ch1_(detection.nim)$', fn)[2])
                    for fn in list(filter(re.compile('.*_ch1_.*').match, filenames))]
        scene_list = [int(re.split('(.*)_([0-9]{2})_s([0-9]+)_ch1_(detection.nim)$', fn)[3])
                      for fn in list(filter(re.compile('.*_ch1_.*').match, filenames))]

    # detect cell centroid
    cell_detection = {}
    area_detection = {}
    for ch in ch_list:
        for slice_idx in list(range(len(scene_list))):
            # for slice_idx in [47]:
            # slice_idx = (file_idx-1)*4 + scene_idx
            file_idx = fid_list[slice_idx]
            scene_idx = scene_list[slice_idx]
            logger.info(f'Running {prefix}_{file_idx:02}_s{scene_idx}_ch{ch}')
            nim_filename = os.path.join(folder, f'{prefix}_{file_idx:02}_s{scene_idx}_ch{ch}_detection.nim')
            correction_folder = os.path.join(folder, 'background_corrected', 'aligned', 'correction', f'{slice_idx + 1}')

            zimg_cell = ZImg(nim_filename)
            cell_img = zimg_cell.data[0][0, 0, :, :].copy()
            cell_img = np.array(cell_img)
            props = measure.regionprops_table(cell_img, properties=('label', 'centroid', 'area'))
            cell_centroids = np.array([props['centroid-0'], props['centroid-1']]).T
            cell_area = props['area']

            if is_reference:
                hj_transform_mat_file = os.path.join(folder, 'Lemur-H_NeuN_VGluT2_SMI99_tforms_1.mat')
                hj_transforms = scipy.io.loadmat(hj_transform_mat_file)
                tfm = hj_transforms['tforms'][slice_idx, 0].copy().astype(np.float64)
                if tfm[0, 0] < 0:
                    tfm[2, 0] -= 2  # no idea why
                des_height = hj_transforms['refSize'][0, 0]
                des_width = hj_transforms['refSize'][0, 1]
                aligned_centroids = cell_centroids[:, [0, 1]] \
                                    - [cell_img.shape[0] / 2, cell_img.shape[1] / 2] + [14040, 10140]
                aligned_centroids = np.matmul(aligned_centroids, tfm[0:2, 0:2]) + tfm[2, 0:2]
            else:
                # slice_folder_idx = slice_idx + 1
                slice_folder_idx = (file_idx - 1) * 4 + scene_idx + 1
                slice_folder = os.path.join(tform_folder, str(slice_folder_idx))
                tform_name = os.path.join(slice_folder, f'{slice_folder_idx}_1.mat')

                if not os.path.exists(tform_name):
                    break
                if group_flip != None:
                    is_flip = 1 in group_flip[slice_folder_idx - 1]
                else:
                    is_flip = False
                src_height = 1690 * 16  # Specific for Hotsauce PV
                src_width = 1245 * 16  # Specific for Hotsauce PV
                if is_flip:
                    cell_centroids[:, 0] = src_height - cell_centroids[:, 0]

                aligned_centroids = pd.DataFrame(data=cell_centroids[:, [0, 1]].copy(), columns=['x', 'y'])
                aligned_centroids = aligned_centroids / 16
                aligned_centroids = ants.apply_transforms_to_points(2, aligned_centroids, tform_name, whichtoinvert=[True])
                aligned_centroids = aligned_centroids * 16
                aligned_centroids = aligned_centroids.to_numpy()

            # Get label image
            label_name = os.path.join(correction_folder, 'image_label.mhd')
            if not os.path.exists(label_name):
                if slice_idx not in cell_detection.keys():
                    cell_detection[slice_idx] = {}
                    area_detection[slice_idx] = {}
                if ch not in cell_detection[slice_idx].keys():
                    cell_detection[slice_idx][ch] = aligned_centroids
                    area_detection[slice_idx][ch] = cell_area
                else:
                    cell_detection[slice_idx][ch] = np.concatenate((cell_detection[slice_idx][ch], aligned_centroids),
                                                                   axis=0)
                    area_detection[slice_idx][ch] = np.concatenate((area_detection[slice_idx][ch], cell_area),
                                                                   axis=0)
                break

            label_ZImg = ZImg(label_name)
            label_img = label_ZImg.data[0][0, 0, :, :].copy()
            region_list = np.unique(label_img)
            region_list = region_list[region_list != 0].astype(np.uint8)

            # Import manual correction tform
            tform_name = os.path.join(correction_folder, 'manual_tforms.mat')
            tform_mat = scipy.io.loadmat(tform_name)
            tform_mat = tform_mat['tform_mat'][0]
            tformed_centroid = (aligned_centroids) / 16

            valid_cells = find_valid_cells(tformed_centroid, label_img, scale_down=1)
            tformed_centroid = tformed_centroid[valid_cells]
            cell_area = cell_area[valid_cells]

            # Apply manual correction tform to cells
            cell_label = label_img[(tformed_centroid[:, 1]).astype('uint16'),
                                   (tformed_centroid[:, 0]).astype('uint16')]

            for region_id in region_list:
                # Find cell in each region label
                valid_cells = cell_label == region_id
                # Rescale transform
                tform = tform_mat[region_id - 1].copy().astype(np.float64)

                # Apply tform to cells
                tformed_centroid[valid_cells, :] = np.matmul(tformed_centroid[valid_cells, :], tform[0:2, 0:2]) + tform[2, 0:2]
                tformed_centroid[valid_cells, :] *= 16

            if slice_idx not in cell_detection.keys():
                cell_detection[slice_idx] = {}
                area_detection[slice_idx] = {}
            if ch not in cell_detection[slice_idx].keys():
                cell_detection[slice_idx][ch] = tformed_centroid
                area_detection[slice_idx][ch] = cell_area
            else:
                cell_detection[slice_idx][ch] = np.concatenate((cell_detection[slice_idx][ch], tformed_centroid),
                                                               axis=0)
                area_detection[slice_idx][ch] = np.concatenate((area_detection[slice_idx][ch], cell_area),
                                                               axis=0)
            logger.info(f'Detected cells/Area : {len(cell_detection[slice_idx][ch])}/{len(area_detection[slice_idx][ch])}')

    cell_detection_int = convert_key_to_int(cell_detection)
    area_detection_int = convert_key_to_int(area_detection)

    result_filename = os.path.join(result_folder, '00_cell.json')
    with open(result_filename, 'w') as fp:
        json.dump(cell_detection_int, fp)
    result_filename = os.path.join(result_folder, '00_cell_size.json')
    with open(result_filename, 'w') as fp:
        json.dump(area_detection_int, fp)


def compute_density(cell_filename: str, ra_filename: str, size_filename: str = None, target_slicelist: list = None, scale_ratio: int = 8,
                    save_figure: bool = False, save_csv: bool = True):
    result_folder = os.path.dirname(cell_filename)

    # Load Cell data
    with open(cell_filename) as json_file:
        cells = json.load(json_file)
    # Load and process cell size data
    if size_filename:
        with open(size_filename) as json_file:
            cells_size = json.load(json_file)

    ch_list = list(cells[list(cells.keys())[0]].keys())

    # Load region annotation file and convert into mask
    scale_down = 1.0 / scale_ratio
    scale_width = 28080 * scale_down
    scale_height = 20280 * scale_down
    dapi_coeff = 1.730
    neun_coeff = 1.864
    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_filename}')
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    logger.info(f'finish reading masks from {ra_filename}')

    # Get lemur region ontology
    ontology = get_ontology()
    id_to_name = dict(zip(region_to_masks.keys(), get_name_from_ontology(ontology, [int(x) for x in list(region_to_masks.keys())])))

    # Initialize variables
    if target_slicelist:
        slice_list = [str(x) for x in target_slicelist]
    else:
        slice_list = list(cells.keys())

    ch_idx = 0
    count_df_list = []
    region_df_list = []
    size_df_list = []
    region_voxel_count = {}
    region_cell_count = {}
    region_cell_size = {}
    for ch_str in ch_list:
        logger.info(f'Counting ch{ch_str}')
        # Create cell and region dataframe
        cell_df = pd.DataFrame(columns=region_to_masks.keys())
        region_df = pd.DataFrame(columns=region_to_masks.keys())
        size_df = pd.DataFrame(columns=region_to_masks.keys())
        ch_cells = None

        for slice_str in slice_list:
            logger.info(f'Counting cells in {slice_str}')
            slice_idx = int(slice_str)
            mask_slice = extract_slice_from_region_mask(region_to_masks, slice_idx, scale_height, scale_width)
            slice_cells = np.array(cells[f'{slice_idx}'][ch_str])
            valid_cells = find_valid_cells(slice_cells, mask_slice, scale_down)
            slice_cells = slice_cells[valid_cells]
            cell_label = mask_slice[(slice_cells[:, 1] * scale_down).astype('uint16'),
                                    (slice_cells[:, 0] * scale_down).astype('uint16')]

            # Count number of cells within unique region in the slice and update the cell number within the region
            unique, counts = np.unique(cell_label, return_counts=True)
            slice_count = dict(zip(unique, counts))
            # Multiply 3D coefficient
            if ch_str == '1':
                slice_count = slice_count * dapi_coeff
            else:
                slice_count = slice_count * neun_coeff
            cell_df = cell_df.append(slice_count, ignore_index=True)
            # Sum of cell size within unique region in the slice and update the summation of cell size within the region
            if size_filename:
                slice_cells_size = np.array(cells_size[f'{slice_idx}'][ch_str])
                slice_cells_size = slice_cells_size[valid_cells]
                slice_size = {key: np.sum([v[1] for v in val]) for key, val in
                              groupby(sorted(list(zip(cell_label, slice_cells_size)), key=lambda ele: ele[0]), key=lambda ele: ele[0])}
                # region_cell_size = {key: region_cell_size.get(key, 0) + slice_size.get(key, 0) for key in set(region_cell_size) | set(slice_size)}
                size_df = size_df.append(slice_size, ignore_index=True)
            # Sum of region size(in voxel) within unique region in the slice and update the total region size
            unique, counts = np.unique(mask_slice, return_counts=True)
            slice_region = dict(zip(unique, counts))
            # Multiply voxel size to get the actual size
            slice_region = slice_region * (0.648 * scale_ratio)^2 * 50
            region_df = region_df.append(slice_region, ignore_index=True)

        cell_df = cell_df.rename(columns=id_to_name, inplace=False)
        cell_df = cell_df.T.fillna(0)
        region_df = region_df.rename(columns=id_to_name, inplace=False)
        region_df = region_df.T.fillna(0)
        size_df = size_df.rename(columns=id_to_name, inplace=False)
        size_df = size_df.T.fillna(0)

        # update ch index
        count_df_list.append(cell_df)
        region_df_list.append(region_df)
        size_df_list.append(size_df)
        ch_idx += 1

    # Visualization density
    # if save_figure:


    # Save result
    if save_csv:
        # Save Cell Count CSV
        for (ch_str, df) in zip(ch_list, count_df_list):
            count_filename = os.path.join(result_folder, f'00_cell_{ch_str}_count.csv')
            df.to_csv(path_or_buf=count_filename, sep=',')
            # Generate region concatenated csv
            concatenate_region_csv(count_filename, ontology, ra_dict)
        # Save Cell Size CSV
        for (ch_str, df) in zip(ch_list, size_df_list):
            size_filename = os.path.join(result_folder, f'00_cell_{ch_str}_size.csv')
            df.to_csv(path_or_buf=size_filename, sep=',')
            # Generate region concatenated csv
            concatenate_region_csv(size_filename, ontology, ra_dict)
        # Save Region Size
        region_filename = os.path.join(result_folder, '00_region_size.csv')
        region_df_list[0].to_csv(path_or_buf=region_filename, sep=',')
        # Generate region concatenated csv
        concatenate_region_csv(region_filename, ontology, ra_dict)

        for (ch_str, df) in zip(ch_list, size_df_list):
            count_filename = os.path.join(result_folder, f'00_cell_{ch_str}_count_region_slice_merged.csv')
            size_filename = os.path.join(result_folder, f'00_cell_{ch_str}_size_region_slice_merged.csv')
            region_filename = os.path.join(result_folder, '00_region_size_region_slice_merged.csv')
            soma_filename = os.path.join(result_folder, f'00_cell_{ch_str}_soma.csv')
            density_filename = os.path.join(result_folder, f'00_cell_{ch_str}_density.csv')
            compute_ratio_csv(size_filename, count_filename, soma_filename)
            compute_ratio_csv(count_filename, region_filename, density_filename)

    return count_df_list, region_df_list, size_df_list


def save_region_cell(cell_filename: str, ra_filename: str, target_slicelist: list = None, region_name: str = None, scale_ratio: int = 8,
                     downsample_ratio:
                     int = 10):
    # Get ontology info
    json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v5.json', 'r')
    ontology = json.load(json_file)
    ontology = ontology['msg'][0]
    json_file.close()
    if region_name:
        child_list = get_region_from_ontology(ontology, get_id_from_ontology(ontology, region_name))
        child_list = child_list['children']
        child_list.append(get_id_from_ontology(ontology, region_name))

    result_folder = os.path.dirname(cell_filename)
    scale_down = 1.0 / scale_ratio

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_filename}')
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    logger.info(f'finish reading masks from {ra_filename}')
    id_to_name = dict(zip(region_to_masks.keys(), get_name_from_ontology(ontology, [int(x) for x in list(region_to_masks.keys())])))

    # Load Cell data
    with open(cell_filename) as json_file:
        cells = json.load(json_file)

    if target_slicelist:
        slice_list = [str(x) for x in target_slicelist]
    else:
        slice_list = list(cells.keys())

    ch_list = list(cells[list(cells.keys())[0]].keys())
    # density_volume = np.zeros(shape=(len(ch_list), 180, int(20280 * scale_down), int(28080 * scale_down)))
    ch_idx = 0
    for ch_str in ch_list:
        logger.info(f'Counting ch{ch_str}')
        # Create cell and region dataframe
        cell_df = pd.DataFrame(columns=region_to_masks.keys())
        region_df = pd.DataFrame(columns=region_to_masks.keys())
        ch_cells = None
        for slice_str in slice_list:
            logger.info(f'Counting cells in {slice_str}')
            slice_idx = int(slice_str)
            mask_slice = extract_slice_from_region_mask(region_to_masks, slice_idx, 20280 * scale_down, 28080 * scale_down)
            slice_cells = np.array(cells[f'{slice_idx}'][ch_str])
            invalid_cells = np.argwhere(np.sum(slice_cells < 0, axis=1))
            slice_cells = np.delete(slice_cells, invalid_cells, axis=0)
            invalid_cells = np.logical_or(slice_cells[:, 1] * scale_down > mask_slice.shape[0],
                                          slice_cells[:, 0] * scale_down > mask_slice.shape[1])
            slice_cells = slice_cells[~invalid_cells]
            if region_name:
                valid_cells = np.isin(mask_slice[np.round(slice_cells[:, 1] * scale_down).astype('uint16'),
                                                 np.round(slice_cells[:, 0] * scale_down).astype('uint16')], child_list)
                slice_cells = slice_cells[valid_cells]

            cell_label = mask_slice[(slice_cells[:, 1] * scale_down).astype('uint16'),
                                    (slice_cells[:, 0] * scale_down).astype('uint16')]
            slice_cells = np.concatenate((slice_cells, np.tile(slice_idx, [slice_cells.shape[0], 1])), axis=1)
            if ch_cells is None:
                ch_cells = slice_cells
            else:
                ch_cells = np.concatenate((ch_cells, slice_cells), axis=0)

        # Save result
        if ch_str == '2':
            cell_downsample = int(downsample_ratio / 2)
        else:
            cell_downsample = downsample_ratio

        result_filename = os.path.join(result_folder, f'00_cell_{ch_str}')
        if region_name:
            result_filename = f'{result_filename}_{region_name}'
        if target_slicelist:
            result_filename = f'{result_filename}_slice{min(target_slicelist)}-{max(target_slicelist)}'
        result_filename = f'{result_filename}_down{cell_downsample}.txt'

        np.savetxt(result_filename, ch_cells[::cell_downsample, :], delimiter=',')

        # update ch index
        ch_idx += 1

    # result_filename = os.path.join(result_folder, f'00_density_region.nim')
    # img_util.write_img(result_filename, density_volume)


def compute_ratio_csv(nom_csv: str, denom_csv: str, result_filename: str):
    nom_df = pd.read_csv(nom_csv).T
    denom_df = pd.read_csv(denom_csv).T
    nom_df = nom_df.rename(columns=nom_df.iloc[0]).drop(nom_df.index[0]).astype('float')
    denom_df = denom_df.rename(columns=denom_df.iloc[0]).drop(denom_df.index[0]).astype('float')

    ratio_df = nom_df.divide(denom_df)
    ratio_df.T.to_csv(path_or_buf=result_filename, sep=',')


def concatenate_region_csv(cell_csv: str, ontology: dict, ra_dict: dict):
    def merge_region_df(ontology: dict, region_name: str, df: pd.DataFrame):
        # Recursively process hierarchical structure
        sum_value = 0
        if region_name == ontology['name']:
            sum_value += df[region_name].copy() if region_name in df.keys() else 0
            for child_ontology in ontology['children']:
                sum_value += merge_region_df(child_ontology, get_name_from_ontology(ontology, child_ontology['id'])[0], df)
        else:
            for child_ontology in ontology['children']:
                sum_value += merge_region_df(child_ontology, region_name, df)
        return sum_value

    cell_df = pd.read_csv(cell_csv).T
    cell_df = cell_df.rename(columns=cell_df.iloc[0]).drop(cell_df.index[0])

    region_list = [x for x in ra_dict['Regions'].keys() if x > 0]
    sum_cell_df = pd.DataFrame(columns=region_list)

    id_to_name = dict(zip(region_list, get_name_from_ontology(ontology, [int(x) for x in region_list])))
    for region_id in sum_cell_df.keys():
        if region_id == -1:
            continue
        region_name = get_name_from_ontology(ontology, region_id)[0]
        sum_cell_df[region_id] = merge_region_df(ontology, region_name, cell_df)

    sum_cell_df = sum_cell_df.rename(columns=id_to_name, inplace=False)
    sum_cell_df = sum_cell_df.fillna(0)

    result_folder = os.path.dirname(cell_csv)
    filename = os.path.basename(cell_csv)

    # Save result
    result_filename = os.path.join(result_folder, f'{os.path.basename(cell_csv)[:-4]}_region_merged.csv')
    sum_cell_df.T.to_csv(path_or_buf=result_filename, sep=',')
    result_filename = os.path.join(result_folder, f'{os.path.basename(cell_csv)[:-4]}_region_slice_merged.csv')
    pd.DataFrame({'sum_slice': sum_cell_df.sum(axis=0)}).to_csv(path_or_buf=result_filename, sep=',')

    return result_filename


def generate_heatmap_csv(cell_csv: str, region_to_masks: dict, result_filename: str = None, slice_list: list = None, scale_ratio: int = 8):
    df = pd.read_csv(cell_csv).T
    df = df.rename(columns=df.iloc[0]).drop(df.index[0])

    scale_down = 1.0 / scale_ratio
    scale_width = int(28080 * scale_down)
    scale_height = int(20280 * scale_down)
    ontology = get_ontology()

    if not slice_list:
        slice_list = list(range(180))
    heatmap_volume = np.zeros(shape=(1, len(slice_list), scale_height, scale_width))
    idx = 0
    for slice_idx in slice_list:
        logger.info(f'Visualizing density in {slice_idx}')
        heatmap_slice = heatmap_volume[0, idx, :, :]
        mask_slice = extract_slice_from_region_mask(region_to_masks, slice_idx, scale_height, scale_width)
        for region_id in np.unique(mask_slice):
            if region_id == 0:
                continue
            else:
                heatmap_slice[mask_slice == region_id] = df[get_name_from_ontology(ontology, [region_id])].to_numpy().sum()
        heatmap_volume[0, idx, :, :] = heatmap_slice
        idx = idx + 1

    if result_filename:
        img_util.write_img(result_filename, heatmap_volume)

    return heatmap_volume


def run_analysis(cells: dict, ra_filename: str, scale_ratio: int = 8):
    scale_down = 1.0 / scale_ratio
    mesh_to_cell_ratio = 16.0 / scale_ratio

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_filename}')
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    logger.info(f'finish reading masks from {ra_filename}')

    region_list = [3851, 3852, 3853, 3854, 3855, 6691, 6692, 6693, 6694, 6695]

    ch_list = list(cells[list(cells.keys())[0]].keys())
    region_cell_count = {}
    for ch in ch_list:
        if ch not in region_cell_count.keys():
            region_cell_count[ch] = {}
        for slice_idx in range(115, 170):
            logger.info(f'running slice {slice_idx}')
            slice_cells = cells[str(slice_idx)][ch]

            for region_id in region_list:
                if region_id not in region_cell_count[ch].keys():
                    region_cell_count[ch][region_id] = {}

                if slice_idx not in region_cell_count[ch][region_id].keys():
                    region_cell_count[ch][region_id][slice_idx] = {}
                    region_cell_count[ch][region_id][slice_idx]['l_count'] = 0
                    region_cell_count[ch][region_id][slice_idx]['r_count'] = 0
                    # region_cell_count[ch][region_id][slice_idx]['l_cell'] = []
                    # region_cell_count[ch][region_id][slice_idx]['r_cell'] = []
                    region_cell_count[ch][region_id][slice_idx]['l_size'] = 0
                    region_cell_count[ch][region_id][slice_idx]['r_size'] = 0

                if slice_idx not in region_to_masks[region_id].keys():
                    continue
                shape = region_to_masks[region_id][slice_idx]
                for subShape in shape:
                    mask, x_start, y_start = subShape[0:3]
                    valid_cells = []
                    for cell in slice_cells:
                        cent = [cell[0] * mesh_to_cell_ratio, cell[1] * mesh_to_cell_ratio]
                        if cent[0] - x_start < 0 or cent[1] - y_start < 0 or cent[0] - x_start >= mask.shape[1] or cent[1] - y_start >= mask.shape[0]:
                            continue
                        is_valid = mask[int(cent[1] - y_start), int(cent[0] - x_start)]
                        if is_valid:
                            valid_cells.append(cell)

                    if x_start > 1755 / 2 * mesh_to_cell_ratio * 0.95:
                        region_cell_count[ch][region_id][slice_idx]['r_count'] += len(valid_cells)
                        # region_cell_count[ch][region_id][slice_idx]['r_cell'].extend(valid_cells)
                        region_cell_count[ch][region_id][slice_idx]['r_size'] += sum(sum(mask))
                    else:
                        region_cell_count[ch][region_id][slice_idx]['l_count'] += len(valid_cells)
                        # region_cell_count[ch][region_id][slice_idx]['l_cell'].extend(valid_cells)
                        region_cell_count[ch][region_id][slice_idx]['l_size'] += sum(sum(mask))

    result_filename = os.path.join(result_folder, '00_cell_visual.json')
    with open(result_filename, 'w') as fp:
        json.dump(region_cell_count, fp, default=np_encoder)


if __name__ == "__main__":
    hyungju_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align'
    ref_folder = os.path.join(hyungju_folder, 'alignment-no-split', 'Hotsauce_SMI99_VGluT2_NeuN')

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

    for idx in [3]:
        result_folder = os.path.join(hyungju_folder, 'alignment-no-split', result_list[idx])

        # ---------------------------------------------------------------------------------------------------------------------------
        # 00. Read group and split info
        # ---------------------------------------------------------------------------------------------------------------------------
        # region_group_name = os.path.join(result_folder, 'region_group.txt')
        # with open(region_group_name) as json_file:
        #     region_group = json.load(json_file)
        #
        # group_split_name = os.path.join(result_folder, 'group_split.txt')
        # with open(group_split_name) as json_file:
        #     group_split = json.load(json_file)
        #
        # reference_group_name = os.path.join(result_folder, 'reference_group.txt')
        # with open(reference_group_name) as json_file:
        #     reference_group = json.load(json_file)
        #
        # template_group_name = os.path.join(ref_folder, 'region_group.txt')
        # with open(template_group_name) as json_file:
        #     template_group = json.load(json_file)
        #
        group_flip_name = os.path.join(result_folder, 'group_flip.txt')
        with open(group_flip_name) as json_file:
            group_flip = json.load(json_file)

        czi_folder = os.path.join(lemur_folder, folder_list[idx])
        tform_folder = os.path.join(result_folder, 'mov')
        register_cells(czi_folder, tform_folder, result_folder, group_flip)
        # cell_filename = os.path.join(result_folder, '00_cell.json')
        # ra_filename = os.path.join(czi_folder, 'matched_reference_annotation.reganno')
        # compute_density(cell_filename, ra_filename, scale_ratio=8)
        # # save_region_cell(cell_filename, ra_filename, 'Thalamus', scale_ratio=16)
        # # save_region_cell(cell_filename, ra_filename, 'CA fields', scale_ratio=8)
