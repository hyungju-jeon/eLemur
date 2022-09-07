import json

import matplotlib.pyplot as pt
import numpy as np
import seaborn as sns

from models.lemur.lemur_cell_counting import *

logger = setup_logger()

def get_clim(img, color_buffer:int = 0.05):
    color_buffer = np.std(img[img>0])
    min_c = np.max([0, np.min(img[img>0])-color_buffer])
    max_c = np.max(img[img>0])*1.01
    return (min_c, max_c)


def save_slice(volume_image:np.ndarray, slice_idx:int, filename:str, clim:tuple=None):
    image_slice = volume_image[0, slice_idx, :, :]
    pt.imshow(image_slice)
    pt.inferno()
    if clim:
        pt.clim(clim)
    else:
        pt.clim(get_clim(image_slice, 0.05))
    pt.colorbar()
    pt.savefig(filename, dpi=600)
    pt.close()


if __name__ == "__main__":
    figure_folder = '/Volumes/shared/Project/InProgress/Mouse_Lemur/7_manuscript/Figure/Figure5/assets'
    data_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/alignment-no-split/'
    lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')

    # Figure 4A
    if False:
        animal_name = 'Hotsauce_PV_TH_NeuN'
        cell_folder = os.path.join(data_folder, animal_name)
        czi_folder = os.path.join(lemur_folder, 'Hotsauce_334A', '181016_Lemur-Hotsauce_PV_TH_NeuN')
        cell_filename = os.path.join(cell_folder, '00_cell.json')
        ra_filename = os.path.join(czi_folder, 'matched_reference_annotation.reganno')

        with open(cell_filename) as json_file:
            cells = json.load(json_file)
        slice_list = list(cells.keys())
        ch_list = list(cells[list(cells.keys())[0]].keys())

        scale_ratio = 8
        json_file = open('/Applications/fenglab/Atlas.app/Contents/Resources/ontology/lemur_atlas_ontology_v3.json', 'r')
        ontology = json.load(json_file)
        ontology = ontology['msg'][0]
        json_file.close()

        result_folder = os.path.dirname(cell_filename)
        scale_down = 1.0 / scale_ratio

        ra_dict = region_annotation.read_region_annotation(ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
        logger.info(f'finish reading {ra_filename}')
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        logger.info(f'finish reading masks from {ra_filename}')
        id_to_name = dict(zip(region_to_masks.keys(), get_name_from_ontology(ontology, [int(x) for x in list(region_to_masks.keys())])))

    # Figure 5B
    if False:
        animal_name = 'Hotsauce_PV_TH_NeuN'
        cell_folder = os.path.join(data_folder, animal_name)
        czi_folder = os.path.join(lemur_folder, 'Hotsauce_334A', '181016_Lemur-Hotsauce_PV_TH_NeuN')
        cell_filename = os.path.join(cell_folder, '00_cell.json')
        ra_filename = os.path.join(czi_folder, 'matched_reference_annotation.reganno')
        # save_region_cell(cell_filename, ra_filename, "Thalamus", scale_ratio= 8)
        # save_region_cell(cell_filename, ra_filename, target_slice = [92, 93, 94], scale_ratio=8, downsample_ratio=10)
        # save_region_cell(cell_filename, ra_filename, scale_ratio=8, downsample_ratio=1)
        save_region_cell(cell_filename, ra_filename, target_slice=[75,76,77,78,79], scale_ratio=8, downsample_ratio=10)
        save_region_cell(cell_filename, ra_filename, target_slice=[67,68,69,70,71], scale_ratio=8, downsample_ratio=10)

    # Figure 5C
    if True:
        animal_name = 'Hotsauce_PV_TH_NeuN'
        cell_folder = os.path.join(data_folder, animal_name)
        czi_folder = os.path.join(lemur_folder, 'Hotsauce_334A', '181016_Lemur-Hotsauce_PV_TH_NeuN')
        cell_filename = os.path.join(cell_folder, '00_cell.json')
        size_filename = os.path.join(cell_folder, '00_cell_size.json')
        ra_filename = os.path.join(czi_folder, 'matched_reference_annotation.reganno')
        # count_df_list, region_df_list, size_df_list = compute_density(cell_filename, ra_filename, size_filename)

        ref_ra_filename = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/reference-data/mesh-high-res/99_final_annotation_for_3d.reganno'
        ra_dict = region_annotation.read_region_annotation(ref_ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * 2)
        logger.info(f'finish reading {ref_ra_filename}')
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        logger.info(f'finish reading masks from {ref_ra_filename}')

        analysis_folder = os.path.join(cell_folder, 'analysis')

        for ch_str in ['1', '2', '4']:
            ratio_csv = os.path.join(analysis_folder, f'00_cell_{ch_str}_density.csv')
            density_filename = os.path.join(analysis_folder, f'01_density_{ch_str}.tiff')
            heatmap_volume = generate_heatmap_csv(ratio_csv, region_to_masks)#, result_filename=density_filename)
            slice_idx = 78
            slice_filename = os.path.join(analysis_folder, f'01_density_{ch_str}_{slice_idx}.pdf')
            save_slice(heatmap_volume, slice_idx, slice_filename)

            ratio_csv = os.path.join(analysis_folder, f'00_cell_{ch_str}_soma.csv')
            size_filename = os.path.join(analysis_folder, f'01_size_{ch_str}.tiff')
            heatmap_size = generate_heatmap_csv(ratio_csv, region_to_masks)#, result_filename=size_filename)
            slice_idx = 78
            slice_filename = os.path.join(analysis_folder, f'01_size_{ch_str}_{slice_idx}.pdf')
            save_slice(heatmap_size, slice_idx, slice_filename)

        dapi_csv = os.path.join(analysis_folder, f'00_cell_1_soma.csv')
        neun_csv = os.path.join(analysis_folder, f'00_cell_4_soma.csv')
        dapi_neun_ratio_csv = os.path.join(analysis_folder, f'02_DAPI_NeuN_size.csv')
        dapi_neun_ratio = os.path.join(analysis_folder, f'02_DAPI_NeuN_size.tiff')
        compute_ratio_csv(dapi_csv, neun_csv, dapi_neun_ratio_csv)
        heatmap_ratio = generate_heatmap_csv(dapi_neun_ratio_csv, region_to_masks, result_filename=dapi_neun_ratio)

        slice_filename = os.path.join(analysis_folder, f'02_DAPI_NeuN_size_{ch_str}_{slice_idx}.pdf')
        save_slice(heatmap_ratio, slice_idx, slice_filename)



