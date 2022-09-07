import json
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

from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.structure_tree import StructureTree
from allensdk.core.reference_space import ReferenceSpace

logger = setup_logger()

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def extract_slice_from_region_mask(region_to_masks: dict, slice_idx: int, height: int, width: int):
    height = int(height)
    width = int(width)
    annotation_mask = np.zeros(shape=(height, width), dtype=np.uint16)
    for region_id, slice_rois in region_to_masks.items():
        if slice_idx in slice_rois.keys():
            maskps = slice_rois[slice_idx]
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
        for i in range(0,len(cell_centroids),10):
            # cent = slice_cells[i]
            cent = cell_centroids[i]
            f.write(f'{cent[0]} {cent[1]}\n')


def register_mouse_cells(folder: str, result_folder:str, brain_info: dict = None):
    result_filename = os.path.join(result_folder, '00_cell.json')
    if os.path.exists(result_filename):
        logger.info(f'Cell mapping result exists')
        return

    (_, _, filenames) = next(os.walk(os.path.join(folder)))
    r = re.compile('.*([0-9]{2})_scene([0-9]+)_ch([0-9]+)_(detection.nim)$')
    filenames = sorted(list(filter(r.match, filenames)))
    prefix = re.split('(.*)_([0-9]{2})_scene([0-9]+)_ch([0-9]+)_(detection.nim)$', filenames[0])[1]
    ch_list = [int(re.split('(.*)_([0-9]{2})_scene([0-9]+)_ch([0-9]+)_(detection.nim)$', fn)[-3]) for fn in filenames]
    ch_list = np.unique(ch_list)

    if brain_info is not None:
        nim_filenames = [os.path.split(x)[1] for x in brain_info['filename']]
        scene_list = [int(x) - 1 for x in brain_info['scene']]
        fid_list = [int(re.split('.*_([0-9]+).czi$', fn)[1]) for fn in nim_filenames]
    else:
        fid_list = [int(re.split('(.*)_([0-9]{2})_scene([0-9]+)_ch1_(detection.nim)$', fn)[2])
                    for fn in list(filter(re.compile('.*_ch1_.*').match, filenames))]
        scene_list = [int(re.split('(.*)_([0-9]{2})_scene([0-9]+)_ch1_(detection.nim)$', fn)[3])
                      for fn in list(filter(re.compile('.*_ch1_.*').match, filenames))]

    # detect cell centroid
    cell_detection = {}
    for ch in ch_list:
        for slice_idx in list(range(len(scene_list))):
            file_idx = fid_list[slice_idx]
            scene_idx = scene_list[slice_idx]
            logger.info(f'Running {prefix}_{file_idx:02}_scece{scene_idx}_ch{ch}')

            nim_filename = os.path.join(folder, f'{prefix}_{file_idx:02}_scene{scene_idx}_ch{ch}_detection.nim')
            zimg_cell = ZImg(nim_filename)
            cell_img = zimg_cell.data[0][0, 0, :, :].copy()
            cell_img = np.array(cell_img)
            props = measure.regionprops_table(cell_img, properties=('label', 'centroid'))
            cell_centroids = np.array([props['centroid-1'], props['centroid-0']]).T


            if slice_idx not in cell_detection.keys():
                cell_detection[slice_idx] = {}
            if ch not in cell_detection[slice_idx].keys():
                cell_detection[slice_idx][ch] = cell_centroids
            else:
                cell_detection[slice_idx][ch] = np.concatenate((cell_detection[slice_idx][ch], cell_centroids),
                                                               axis=0)

            np.savetxt(os.path.join(result_folder, f'{prefix}_{file_idx:02}_scene{scene_idx}_ch{ch}.txt'),
                       np.concatenate((cell_centroids[::10, :], np.tile(slice_idx, [cell_centroids[::10, :].shape[0],1])), axis=1),
                       delimiter=',')
    cell_detection_int = {}
    for slice_idx, cells in cell_detection.items():
        for ch, cell in cells.items():
            if slice_idx not in cell_detection_int.keys():
                cell_detection_int[int(slice_idx)] = {}
            if ch not in cell_detection_int[int(slice_idx)].keys():
                cell_detection_int[int(slice_idx)][int(ch)] = cell.tolist()

    with open(result_filename, 'w') as fp:
        json.dump(cell_detection_int, fp)


def mouse_compute_density(cell_filename: str, annotation_filename: str, scale_ratio: int = 40):
    # Get ontology info
    oapi = OntologiesApi()
    structure_graph = oapi.get_structures_with_sets([1])  # 1 is the id of the adult mouse structure graph
    structure_graph = StructureTree.clean_structures(structure_graph)
    tree = StructureTree(structure_graph)

    result_folder = os.path.dirname(cell_filename)
    scale_down = 1.0 / scale_ratio

    # Load Cell data
    with open(cell_filename) as json_file:
        cells = json.load(json_file)

    slice_list = list(cells.keys())
    ch_list = list(cells[list(cells.keys())[0]].keys())
    region_list = list(tree.get_name_map().values())

    annotation_volume_ZImg = ZImg(annotation_filename)
    annotation_volume = annotation_volume_ZImg.data[0]
    annotation_volume = np.squeeze(annotation_volume)

    # density_volume = np.zeros(shape=(len(ch_list), 180, int(20280 * scale_down), int(28080 * scale_down)))
    ch_idx = 0

    cell_all = dict(zip(ch_list,[pd.DataFrame(columns=region_list)] * len(ch_list)))
    region_all = dict(zip(ch_list,[pd.DataFrame(columns=region_list)] * len(ch_list)))

    for slice_str in slice_list:
        logger.info(f'Counting cells in {slice_str}')
        slice_idx = int(slice_str)
        mask_generator = ReferenceSpace(tree, annotation_volume[slice_idx,:,:], [100, 25, 25])

        slice_cells_ch = {}
        for ch_str in ch_list:
            slice_cells = np.array(cells[f'{slice_idx}'][ch_str])
            invalid_cells = np.argwhere(np.sum((slice_cells < 0), axis=1))
            slice_cells = np.delete(slice_cells, invalid_cells, axis=0)
            invalid_cells = np.argwhere(np.logical_or(slice_cells[:, 1] * scale_down > annotation_volume.shape[1],
                                                      slice_cells[:, 0] * scale_down > annotation_volume.shape[2]))
            slice_cells = np.delete(slice_cells, invalid_cells, axis=0)

            slice_cells_ch[ch_str] = slice_cells

            slice_region = mask_generator.remove_unassigned()
            slice_region_list = [x['name'] for x in slice_region]

        for region_name in slice_region_list:
            mask_id = tree.get_structures_by_name([region_name])[0]['id']
            mask_slice = mask_generator.make_structure_mask([mask_id])

            for ch_str in ch_list:
                slice_cells = slice_cells_ch[ch_str]
                cell_label = mask_slice[(slice_cells[:, 1] * scale_down).astype('uint16'),
                                        (slice_cells[:, 0] * scale_down).astype('uint16')]

                if len(cell_all[ch_str]) < slice_idx + 1:
                    cell_all[ch_str] = cell_all[ch_str].append({region_name : np.sum(cell_label)}, ignore_index=True)
                    region_all[ch_str] = region_all[ch_str].append({region_name : np.sum(mask_slice)}, ignore_index=True)
                else:
                    cell_all[ch_str].iloc[slice_idx][region_name] = np.sum(cell_label)
                    region_all[ch_str].iloc[slice_idx][region_name] = np.sum(mask_slice)

    # Visualization density
    if False:
        density_df = (cell_df.sum(axis=1) / region_df.sum(axis=1))
        for slice_str in slice_list:
            logger.info(f'Visualizing density in {slice_str}')
            slice_idx = int(slice_str)
            density_slice = density_volume[ch_idx, slice_idx, :, :]
            mask_slice = extract_slice_from_region_mask(region_to_masks, slice_idx, 20280 * scale_down, 28080 * scale_down)
            for region_id in np.unique(mask_slice):
                if region_id == 0:
                    continue
                else:
                    density_slice[mask_slice == region_id] = density_df[get_name_from_ontology(ontology, [region_id])]
            density_volume[ch_idx, slice_idx, :, :] = density_slice

    # Save result
    for ch_str in ch_list:
        cell_all[ch_str] = (cell_all[ch_str].T).fillna(0)
        region_all[ch_str] = (region_all[ch_str].T).fillna(0)

        cell_filename = os.path.join(result_folder, f'00_cell_{ch_str}_count.csv')
        cell_all[ch_str].to_csv(path_or_buf=cell_filename, sep=',')
        region_filename = os.path.join(result_folder, '00_region_size.csv')
        region_all[ch_str].to_csv(path_or_buf=region_filename, sep=',')

        # process_cell_csv(cell_filename, region_filename, ontology, ra_dict)


def process_cell_csv(cell_csv: str, region_csv: str, ontology: dict, ra_dict: dict):
    def merge_region_df(ontology: dict, region_name: str, df: pd.DataFrame):
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
    region_df = pd.read_csv(region_csv).T

    cell_df = cell_df.rename(columns=cell_df.iloc[0]).drop(cell_df.index[0])
    region_df = region_df.rename(columns=region_df.iloc[0]).drop(region_df.index[0])

    region_list = [x for x in ra_dict['Regions'].keys() if x > 0]
    sum_cell_df = pd.DataFrame(columns=region_list)
    sum_region_df = pd.DataFrame(columns=region_list)

    id_to_name = dict(zip(region_list, get_name_from_ontology(ontology,[int(x) for x in region_list])))
    for region_id in sum_cell_df.keys():
        if region_id == -1:
            continue
        region_name = get_name_from_ontology(ontology, region_id)[0]
        sum_cell_df[region_id] = merge_region_df(ontology, region_name, cell_df)

    for region_id in sum_cell_df.keys():
        if region_id == -1:
            continue
        region_name = get_name_from_ontology(ontology, region_id)[0]
        sum_region_df[region_id] = merge_region_df(ontology, region_name, region_df)

    sum_cell_df = sum_cell_df.rename(columns=id_to_name, inplace=False)
    sum_region_df = sum_region_df.rename(columns=id_to_name, inplace=False)
    sum_cell_df = (sum_cell_df).fillna(0)
    sum_region_df = (sum_region_df).fillna(0)

    result_folder = os.path.dirname(cell_csv)
    filename = os.path.basename(cell_csv)
    # Save result
    result_filename = os.path.join(result_folder, f'{os.path.basename(cell_csv)[:-4]}_fixed.csv')
    sum_cell_df.to_csv(path_or_buf=result_filename, sep=',')
    result_filename = os.path.join(result_folder, f'{os.path.basename(region_csv)[:-4]}_fixed.csv')
    sum_region_df.to_csv(path_or_buf=result_filename, sep=',')


if __name__ == "__main__":
    detection_folder = '/Users/hyungju/Desktop/hyungju/Data/mouse_cell/aligned_detection'
    annotation_folder = os.path.join(io.jinny_nas_dir(), 'Project', 'InProgress', 'E_I', '1_Serial_data', '3M', 'JK979#1')
    annotation_filename = os.path.join(annotation_folder, 'anno_template_25.nim')
    result_folder = '/Users/hyungju/Desktop/hyungju/Data/mouse_cell/aligned_detection_cell'

    # hyungju_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align'
    # ref_folder = os.path.join(hyungju_folder, 'alignment-no-split', 'Hotsauce_SMI99_VGluT2_NeuN')
    #
    # lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')
    # folder = os.path.join(lemur_folder, 'Hotsauce_334A', '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
    # img_filename = os.path.join(folder, 'hj_aligned', 'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    # midline_filename = os.path.join(folder, 'interns_edited_results', 'sh_cut_in_half.reganno')

    # max slice = 113
    register_mouse_cells(detection_folder, result_folder)
    cell_filename = os.path.join(result_folder, '00_cell.json')
    mouse_compute_density(cell_filename, annotation_filename, scale_ratio= 40)

    # czi_folder = os.path.join(lemur_folder, folder_list[idx])
    # tform_folder = os.path.join(result_folder, 'mov')
    # register_mouse_cells(czi_folder, tform_folder, result_folder, group_flip)
    # cell_filename = os.path.join(result_folder, '00_cell.json')
    # ra_filename = os.path.join(czi_folder, 'matched_eLemur_annotation.reganno')
    # compute_density(cell_filename, ra_filename, scale_ratio=8)

    # {'Isocortex', 'HippocampalFormation', 'RemainingRegion', 'Striatum', 'Pallidum', 'Midbrain', 'Thalamus', 'Hypothalamus', 'Pons', 'Medulla', 'Cerebellum', 'fiberTracts'};

