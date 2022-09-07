from pathlib import PurePath
import multiprocessing
import traceback
import json
import math
import shutil
import numbers
import scipy.ndimage

from zimg import *
import utils.io as io
import utils.region_annotation as region_annotation
from utils.logger import setup_logger
import utils.img_util as img_util



logger = setup_logger()


def export_grouped_label_img(img_filename: str, ra_filename: str,
                             region_group: list, group_split: list,
                             result_filename: str,
                             *, midline_filename: str = None, midline_x: float = None, downsample_ratio: int = 16):
    read_ratio = downsample_ratio
    scale_down = 1.0 / read_ratio  # otherwise the mask will be too big

    infoList = ZImg.readImgInfos(img_filename)
    assert len(infoList) == 1 and infoList[0].numTimes == 1
    img_info = infoList[0]
    logger.info(f'image {infoList[0]}')

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

    annotation_mask = np.zeros(shape=(img_info.depth,
                                      int(math.ceil(img_info.height * scale_down)),
                                      int(math.ceil(img_info.width * scale_down))),
                               dtype=np.uint8)
    for region_id, slice_rois in region_to_masks.items():
        if region_id not in region_id_set:
            continue
        for img_slice_, maskps in slice_rois.items():
            assert img_slice_ % 2 == 0, img_slice_
            img_slice = int(img_slice_ / 2) - 7
            mapped_region_id = 0
            for group_idx, group in enumerate(region_group[img_slice], start=1):
                if region_id in group:
                    mapped_region_id = group_idx
                    break
            if mapped_region_id == 0:
                mapped_region_id = len(region_group[img_slice]) + 1
            need_split = mapped_region_id in group_split[img_slice]
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                mask = np.zeros(shape=(annotation_mask.shape[-2], annotation_mask.shape[-1]), dtype=np.bool)
                mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = compact_mask
                if need_split:
                    if midline_filename is not None:
                        midline_slice_rois = midline_to_masks[-1]
                        assert img_slice in midline_slice_rois, img_slice
                        midline_maskps = midline_slice_rois[img_slice]
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
                            annotation_mask[img_slice][mask] = mapped_region_id + 10
                        else:
                            annotation_mask[img_slice][mask] = mapped_region_id
                    else:
                        assert midline_x is not None
                        if x_start + compact_mask.shape[1] / 2 > midline_x * scale_down:  # right side
                            annotation_mask[img_slice][mask] = mapped_region_id + 10
                        else:
                            annotation_mask[img_slice][mask] = mapped_region_id
                else:
                    annotation_mask[img_slice][mask] = mapped_region_id

    img_util.write_img(result_filename, annotation_mask)


def main():
    # list (slice) of list (groups) of list of number (regions in group)
    region_group = [
        [[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11], [12]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[13], [11]],
[[14], [11]],
[[13], [14], [11]],
[[13], [14], [11]],
[[13], [14], [11]],
[[13], [14], [11]],
[[13], [14], [11]],
[[13], [14], [11]],
[[13], [14], [11]],
[[13], [14], [11]],
[[13], [14], [11]],
[[14], [11]],
[[14], [11]],
[[14], [11]],
[[14], [11]],
[[14], [11]],
[[14], [11]],
[[14], [11]],
[[14], [11]],
[[15], [14], [11]],
[[15], [14], [11]],
[[15], [14], [11]],
[[11], [15], [14]],
[[11], [15], [14]],
[[11], [15], [14]],
[[11], [15], [14]],
[[11], [15], [14]],
[[11], [15], [14]],
[[11], [15], [14]],
[[11], [15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
[[15], [14]],
]

    # list (slice) of list of number (groups to be split)
    group_split = [
        [-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
    ]

    lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')
    folder = os.path.join(lemur_folder, 'Hotsauce_334A', '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
    img_filename = os.path.join(folder, 'hj_aligned', 'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    midline_filename = os.path.join(folder, 'interns_edited_results', 'sh_cut_in_half.reganno')

    ra_filename = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/shifted_Hotsauce_blockface-outline_grouped_fix_interpolated.reganno'
    #ra_filename = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/edited_merge_20201001_2.reganno'
    #result_filename = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/edited_merge_20201001_2_merged_label_2.nim'
    result_filename = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/shifted_Hotsauce_blockface-outline_grouped_fix_interpolated.nim'
    #export_grouped_label_img(img_filename, ra_filename, region_group, group_split,
    #                         result_filename, midline_filename=midline_filename)
    export_grouped_label_img(img_filename, ra_filename, region_group, group_split,
                             result_filename, midline_x=28080 / 2.0)

    print('done')


if __name__ == "__main__":
    main()
