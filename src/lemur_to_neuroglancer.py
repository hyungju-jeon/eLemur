from pathlib import PurePath
import multiprocessing
import traceback
import json
import math
import shutil
import numbers

import numpy as np
from zimg import *
import utils.io as io
import utils.region_annotation as region_annotation
from utils.logger import setup_logger
import utils.img_util as img_util

from neuroglancer_scripts.dyadic_pyramid import fill_scales_for_dyadic_pyramid
import tensorstore as ts
from tqdm import tqdm, trange

logger = setup_logger()


def lemur_to_neuroglancer(task: dict):
    if not os.path.exists(task['out_folder']):
        os.mkdir(task['out_folder'])
    sharding = {'@type': 'neuroglancer_uint64_sharded_v1',
                'hash': 'identity',
                'minishard_bits': 6,
                'minishard_index_encoding': 'gzip',
                'data_encoding': 'gzip',
                'preshift_bits': 9,
                'shard_bits': 15}

    if 'img_name' in task:
        img_scale = task['img_scale'] if 'img_scale' in task else 1.
        img_resolution = task['img_resolution'] if 'img_resolution' in task else task['resolution']
        infoList = ZImg.readImgInfos(task['img_name'])
        assert len(infoList) == 1 and infoList[0].numTimes == 1
        img_info = infoList[0]
        logger.info(f'image {infoList[0]}')

        # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md
        img_data_type = img_info.dataTypeString()
        convert_img_to_float32 = img_data_type == 'float64'
        convert_img_to_uint8 = False
        if not convert_img_to_float32:
            if img_data_type not in ["uint8", "uint16", "uint32", "uint64", "float32"]:
                convert_img_to_uint8 = True
                img_data_type = 'uint8'
        else:
            img_data_type = 'float32'

        if img_data_type == 'uint8':
            img_data_range = [0, np.iinfo(np.uint8).max]
        elif img_data_type == 'uint16':
            img_data_range = [0, np.iinfo(np.uint16).max]
        elif img_data_type == 'uint32':
            img_data_range = [0, np.iinfo(np.uint32).max]
        elif img_data_type == 'uint64':
            img_data_range = [0, np.iinfo(np.uint64).max]
        elif img_data_type == 'float32':
            img_data_range = [0.0, 1.0]
        else:
            img_data_range = [0.0, 1.0]
            assert False, img_data_type

        base_info = {
            "dimensions": {
                "x": [
                    img_resolution[0] / img_scale * 1e-9,
                    "m"
                ],
                "y": [
                    img_resolution[1] / img_scale * 1e-9,
                    "m"
                ],
                "z": [
                    img_resolution[2] * 1e-9,
                    "m"
                ]
            },
            "position": [
                img_info.width / 2.0 * img_scale,
                img_info.height / 2.0 * img_scale,
                img_info.depth / 2.0,
            ],
            "crossSectionScale": 59.90389939556908,
            "projectionOrientation": [
                -0.11555982381105423,
                -0.09716008603572845,
                0.4296676218509674,
                0.8902761340141296
            ],
            "projectionScale": 50764.49878262887,
            "layers": [],
            "selectedLayer": {
                "layer": task['annotation_tab_name'] if 'annotation_tab_name' in task else "annotation",
                "visible": True,
            },
            "layout": "4panel",
        }

        assert len(task['channel_names']) == img_info.numChannels

        start_ch = 1e10
        end_ch = -1
        chs = []
        ch_layers = {}
        for ch in range(img_info.numChannels):
            channel_name = task['channel_names'][ch]
            ch_layers[ch] = {
                "type": "image",
                "source": f"precomputed://https://eeum-brain.com/static/neuroglancer_data/{task['out_folder_name']}/{channel_name}",
                "opacity": 1,
                "blend": "additive",
                "shaderControls": {
                    "color": f'#{img_info.channelColors[ch].r:02x}{img_info.channelColors[ch].g:02x}{img_info.channelColors[ch].b:02x}',
                    "normalized": {
                        "range": img_data_range
                    },
                },
                "name": channel_name
            }
            base_info['layers'].append(ch_layers[ch])

            channel_out_folder = os.path.join(task['out_folder'], f"{channel_name}")
            if not os.path.exists(channel_out_folder):
                start_ch = min(start_ch, ch)
                end_ch = max(end_ch, ch)
                chs.append(ch)

        if len(chs) > 0:
            ch_img_data_range = {}
            for ch in chs:
                if img_data_type == 'float32':
                    ch_img_data_range[ch] = [np.finfo(np.float32).max, np.finfo(np.float32).min]
                else:
                    ch_img_data_range[ch] = [np.iinfo(np.uint64).max, np.iinfo(np.uint64).min]

            fullres_info = {}
            fullres_info['type'] = 'image'
            fullres_info['data_type'] = img_data_type
            fullres_info['num_channels'] = 1
            fullres = {}
            fullres['chunk_sizes'] = []
            fullres['encoding'] = 'raw'
            # fullres['jpeg_quality'] = 90
            fullres['sharding'] = sharding
            fullres['resolution'] = [img_resolution[0] / img_scale, img_resolution[1] / img_scale, img_resolution[2]]
            fullres['size'] = [math.ceil(img_info.width * img_scale), math.ceil(img_info.height * img_scale),
                               img_info.depth]
            fullres['voxel_offset'] = [0, 0, 0]
            fullres_info['scales'] = [fullres, ]

            fill_scales_for_dyadic_pyramid(fullres_info, target_chunk_size=64)

            for scale in fullres_info['scales']:
                scale['chunk_size'] = scale['chunk_sizes'][0]
                del scale['chunk_sizes']
                del scale['key']
                arrs = {}
                for ch in chs:
                    channel_name = task['channel_names'][ch]
                    spec = {'driver': 'neuroglancer_precomputed',
                            'kvstore': {'driver': 'file', 'path': task['out_folder']},
                            'path': channel_name,
                            # 'context': {
                            #     'cache_pool': {
                            #         'total_bytes_limit': 1_000_000_000
                            #     }
                            # },
                            'scale_metadata': scale,
                            'multiscale_metadata': {'data_type': fullres_info['data_type'],
                                                    'num_channels': fullres_info['num_channels'],
                                                    'type': fullres_info['type']
                                                    }
                            }
                    arrs[ch] = ts.open(spec=spec, open=True, create=True).result()

                save_step = scale['chunk_size'][2] * 2
                slices = [slice(idx * save_step, min(img_info.depth, save_step * (idx + 1)), 1) for idx in
                          range((img_info.depth + save_step - 1) // save_step)]

                xRatio = round(scale['resolution'][0] / fullres['resolution'][0] / img_scale)
                yRatio = round(scale['resolution'][1] / fullres['resolution'][1] / img_scale)
                zRatio = round(scale['resolution'][2] / fullres['resolution'][2])
                for sl in tqdm(slices):
                    img = ZImg(task['img_name'],
                               region=ZImgRegion(ZVoxelCoordinate(0, 0, sl.start, start_ch, 0),
                                                 ZVoxelCoordinate(-1, -1, sl.stop, end_ch + 1, 1)),
                               scene=0,
                               xRatio=xRatio,
                               yRatio=yRatio,
                               zRatio=zRatio
                               )
                    target_z_start = (sl.start + zRatio - 1) // zRatio
                    target_z_stop = (sl.stop + zRatio - 1) // zRatio
                    for ch in chs:
                        ch_img_data = img.data[0][ch - start_ch]
                        if convert_img_to_float32:
                            assert ch_img_data.dtype == np.float64, ch_img_data.dtype
                            ch_img_data = ch_img_data.astype(np.float32)
                        elif convert_img_to_uint8:
                            assert issubclass(ch_img_data.dtype, numbers.Integral), ch_img_data.dtype
                            ch_img_data = (ch_img_data.astype(np.float64) - np.iinfo(ch_img_data.dtype).min * 1.0) / \
                                          (np.iinfo(ch_img_data.dtype).max * 1.0 - np.iinfo(
                                              ch_img_data.dtype).min * 1.0)

                        if xRatio == 1 and yRatio == 1 and zRatio == 1:
                            ch_img_data_range[ch][0] = min(ch_img_data_range[ch][0], ch_img_data.min())
                            ch_img_data_range[ch][1] = max(ch_img_data_range[ch][1], ch_img_data.max())
                        arrs[ch][ts.d['channel'][0]][ts.d['z'][target_z_start:target_z_stop]] = np.reshape(
                            np.ravel(ch_img_data, order='C'),
                            (ch_img_data.shape[-1], ch_img_data.shape[-2], ch_img_data.shape[-3]),
                            order='F')

            for ch in chs:
                ch_layers[ch]['shaderControls']['normalized']['range'] = [int(ch_img_data_range[ch][0]),
                                                                          int(ch_img_data_range[ch][1])]

    if 'annotation_name' in task:
        annotation_scale = task['annotation_scale'] if 'annotation_scale' in task else 1.
        annotation_resolution = task['annotation_resolution'] if 'annotation_resolution' in task else task['resolution']
        if 'img_name' in task:
            annotation_depth = img_info.depth
            annotation_height = img_info.height
            annotation_width = img_info.width
        else:
            annotation_depth = task['annotation_depth']
            annotation_height = task['annotation_height']
            annotation_width = task['annotation_width']
            base_info = {
                "dimensions": {
                    "x": [
                        annotation_resolution[0] / annotation_scale * 1e-9,
                        "m"
                    ],
                    "y": [
                        annotation_resolution[1] / annotation_scale * 1e-9,
                        "m"
                    ],
                    "z": [
                        annotation_resolution[2] * 1e-9,
                        "m"
                    ]
                },
                "position": [
                    annotation_width / 2.0 * annotation_scale,
                    annotation_height / 2.0 * annotation_scale,
                    annotation_depth / 2.0,
                ],
                "crossSectionScale": 3.74,
                "projectionOrientation": [
                    0.10651738196611404, 0.7164360880851746, -0.046973973512649536, -0.6878712177276611
                ],
                "projectionScale": 3173,
                "layers": [],
                "showSlices": False,
                "selectedLayer": {
                    "layer": task['annotation_tab_name'] if 'annotation_tab_name' in task else "annotation",
                    "visible": True,
                },
                "layout": "3d",
            }

        folder_name = 'annotation'
        annotation_layer = {
            "type": "segmentation",
            "source": f"precomputed://https://eeum-brain.com/static/neuroglancer_data/{task['out_folder_name']}/{folder_name}",
            "tab": "segments",
            "segmentColors": {},
            "name": task['annotation_tab_name'] if 'annotation_tab_name' in task else "annotation",
            "segments": [],
        }

        ra_dict = region_annotation.read_region_annotation(task['annotation_name'])
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * annotation_scale)
        logger.info(f"finish reading {task['annotation_name']}")
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        logger.info(f"finish reading masks from {task['annotation_name']}")

        region_id_set = set()
        # for region_id, slice_rois in region_to_masks.items():
        #     if region_id < 0:
        #         continue
        #     for img_slice, maskps in slice_rois.items():
        #         for compact_mask, x_start, y_start, _ in maskps:
        #             if compact_mask.sum() == 0:
        #                 continue
        #             region_id_set.add(region_id)
        for region_id, region_props in ra_dict['Regions'].items():
            if region_id < 0:
                continue
            if region_props['ROI'] is not None:
                region_id_set.add(region_id)

        if 'annotation_exclude_regions' in task:
            for region_id in task['annotation_exclude_regions']:
                if region_id in region_id_set:
                    region_id_set.remove(region_id)
        for region_id in region_id_set:
            if 'annotation_hide_regions' in task and region_id not in task['annotation_hide_regions']:
                annotation_layer['segments'].append(region_id)

        for region_id, region_props in ra_dict['Regions'].items():
            if region_id in region_id_set:
                annotation_layer['segmentColors'][str(region_id)] = \
                    f"#{region_props['Color'][0]:02x}{region_props['Color'][1]:02x}{region_props['Color'][2]:02x}"

        base_info['layers'].append(annotation_layer)

        annotation_out_folder = os.path.join(task['out_folder'], folder_name)
        if not os.path.exists(annotation_out_folder):
            os.mkdir(annotation_out_folder)

            annotation_mask = np.zeros(shape=(annotation_depth,
                                              int(math.ceil(annotation_height * annotation_scale)),
                                              int(math.ceil(annotation_width * annotation_scale))),
                                       dtype=np.uint32)
            for region_id, slice_rois in region_to_masks.items():
                if region_id not in region_id_set:
                    continue
                for img_slice, maskps in slice_rois.items():
                    for compact_mask, x_start, y_start, _ in maskps:
                        if compact_mask.sum() == 0:
                            continue
                        assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                        mask = np.zeros(shape=(annotation_mask.shape[-2], annotation_mask.shape[-1]), dtype=np.bool_)
                        mask[y_start:y_start + compact_mask.shape[0],
                        x_start:x_start + compact_mask.shape[1]] = compact_mask
                        annotation_mask[img_slice][mask] = region_id

            fullres_info = {}
            fullres_info['type'] = 'segmentation'
            fullres_info['data_type'] = 'uint32'
            fullres_info['num_channels'] = 1
            fullres = {}
            fullres['chunk_sizes'] = []
            fullres['encoding'] = 'compressed_segmentation'
            fullres['compressed_segmentation_block_size'] = [8, 8, 8]
            fullres['sharding'] = sharding
            fullres['resolution'] = [annotation_resolution[0] / annotation_scale,
                                     annotation_resolution[1] / annotation_scale,
                                     annotation_resolution[2]]
            fullres['size'] = [annotation_mask.shape[-1], annotation_mask.shape[-2], annotation_mask.shape[-3]]
            fullres['voxel_offset'] = [0, 0, 0]
            fullres_info['scales'] = [fullres, ]

            fill_scales_for_dyadic_pyramid(fullres_info, target_chunk_size=64)

            for scale in fullres_info['scales']:
                scale['chunk_size'] = scale['chunk_sizes'][0]
                del scale['chunk_sizes']
                del scale['key']
                spec = {'driver': 'neuroglancer_precomputed',
                        'kvstore': {'driver': 'file', 'path': task['out_folder']},
                        'path': folder_name,
                        # 'context': {
                        #     'cache_pool': {
                        #         'total_bytes_limit': 1_000_000_000
                        #     }
                        # },
                        'scale_metadata': scale,
                        'multiscale_metadata': {'data_type': fullres_info['data_type'],
                                                'num_channels': fullres_info['num_channels'],
                                                'type': fullres_info['type']
                                                }
                        }
                arr = ts.open(spec=spec, open=True, create=True).result()
                xRatio = round(scale['resolution'][0] / fullres['resolution'][0])
                yRatio = round(scale['resolution'][1] / fullres['resolution'][1])
                zRatio = round(scale['resolution'][2] / fullres['resolution'][2])
                annotation_mask_for_current_scale = \
                    img_util.imresize(annotation_mask,
                                      des_depth=(annotation_mask.shape[-3] + zRatio - 1) // zRatio,
                                      des_height=(annotation_mask.shape[-2] + yRatio - 1) // yRatio,
                                      des_width=(annotation_mask.shape[-1] + xRatio - 1) // xRatio,
                                      interpolant=Interpolant.Nearest
                                      )
                arr[ts.d['channel'][0]] = np.reshape(
                    np.ravel(annotation_mask_for_current_scale, order='C'),
                    (annotation_mask_for_current_scale.shape[-1],
                     annotation_mask_for_current_scale.shape[-2],
                     annotation_mask_for_current_scale.shape[-3]),
                    order='F')

            with open(os.path.join(annotation_out_folder, 'info'), 'r') as annotation_info_file:
                annotation_info = json.load(annotation_info_file)

            annotation_info["segment_properties"] = "segment_properties"
            annotation_sp_folder = os.path.join(annotation_out_folder, annotation_info["segment_properties"])
            os.mkdir(annotation_sp_folder)
            segment_properties_info = {
                "@type": "neuroglancer_segment_properties",
                "inline": {
                    "ids": [],
                    "properties": [
                        {
                            "id": "label",
                            "type": "label",
                            "values": []
                        },
                    ]
                }
            }
            for region_id, region_props in ra_dict['Regions'].items():
                if region_id in region_id_set:
                    segment_properties_info['inline']['ids'].append(str(region_id))
                    segment_properties_info['inline']['properties'][0]['values'].append(
                        f"{region_props['Name'].replace('/','_')} ({region_props['Abbreviation']})")
            with open(os.path.join(annotation_sp_folder, 'info'), 'w') as outfile:
                json.dump(segment_properties_info, outfile)

            if 'annotation_mesh_folder' in task:
                annotation_mesh_resolution = task['annotation_mesh_resolution'] if 'annotation_mesh_resolution' in task \
                    else task['resolution']
                annotation_info["mesh"] = "mesh"
                annotation_mesh_folder = os.path.join(annotation_out_folder, annotation_info["mesh"])
                os.mkdir(annotation_mesh_folder)
                for region_id, region_props in ra_dict['Regions'].items():
                    if region_id in region_id_set:
                        mesh_filename = os.path.join(task['annotation_mesh_folder'],
                                                     f"{region_props['Name'].replace('/','_')}.obj")
                        print(mesh_filename)
                        mesh = ZMesh(mesh_filename)
                        mesh.vertices *= np.array([annotation_mesh_resolution[0],
                                                   annotation_mesh_resolution[1],
                                                   annotation_mesh_resolution[2]],
                                                  dtype=np.float32)
                        mesh.save(os.path.join(annotation_mesh_folder,
                                               f"{region_props['Name'].replace('/','_')}.precomputed_mesh"),
                                  'precomputed')
                        with open(os.path.join(annotation_mesh_folder, f'{str(region_id)}:0'), 'w') as outfile:
                            json.dump({"fragments": [f"{region_props['Name'].replace('/','_')}.precomputed_mesh"]}, outfile)
                with open(os.path.join(annotation_mesh_folder, 'info'), 'w') as outfile:
                    json.dump({"@type": "neuroglancer_legacy_mesh"}, outfile)

            with open(os.path.join(annotation_out_folder, 'info'), 'w') as outfile:
                json.dump(annotation_info, outfile)

    if 'cell_annotations' in task:
        cell_annotation_resolution = task['cell_annotation_resolution'] if 'cell_annotation_resolution' in task \
            else task['resolution']
        import struct
        for cell_annotaion, cell_annotation_filename in task['cell_annotations'].items():
            folder_name = cell_annotaion
            cell_annotation_layer = {
                "type": "annotation",
                "source": f"precomputed://https://eeum-brain.com/static/neuroglancer_data/{task['out_folder_name']}/{folder_name}",
                "tab": "rendering",
                "name": cell_annotaion
            }

            base_info['layers'].append(cell_annotation_layer)
            cell_annotation_out_folder = os.path.join(task['out_folder'], folder_name)
            if not os.path.exists(cell_annotation_out_folder):
                os.mkdir(cell_annotation_out_folder)

                cell_annotation_array = np.load(cell_annotation_filename)
                x_max = np.ceil(cell_annotation_array[:, 0].max()) + 1
                y_max = np.ceil(cell_annotation_array[:, 1].max()) + 1
                z_max = np.ceil(cell_annotation_array[:, 2].max()) + 1
                lower_bound = [0, 0, 0]

                cell_annotation_info = {
                    "@type": "neuroglancer_annotations_v1",
                    "annotation_type": "POINT",
                    "by_id": {
                        "key": "by_id"
                    },
                    "dimensions": {
                        "x": [
                            cell_annotation_resolution[0] * 1e-9,
                            "m"
                        ],
                        "y": [
                            cell_annotation_resolution[1] * 1e-9,
                            "m"
                        ],
                        "z": [
                            cell_annotation_resolution[2] * 1e-9,
                            "m"
                        ]
                    },
                    "lower_bound": lower_bound,
                    "properties": [],
                    "relationships": [],
                    "spatial": [],
                    "upper_bound": [x_max, y_max, z_max],
                }

                cell_annotation_array = np.concatenate((cell_annotation_array,
                                                        np.arange(cell_annotation_array.shape[0])[:, np.newaxis]),
                                                       axis=1)
                np.random.shuffle(cell_annotation_array)

                level = 0
                remaining_annotations = cell_annotation_array
                level_resolution = cell_annotation_resolution
                chunk_size = [x_max, y_max, z_max]
                grid_shape = [1, 1, 1]
                limit = 100000
                while remaining_annotations.size > 0:
                    key = f'spatial{level}'
                    level_cell_annotation_out_folder = os.path.join(cell_annotation_out_folder, key)
                    if not os.path.exists(level_cell_annotation_out_folder):
                        os.mkdir(level_cell_annotation_out_folder)
                    remaining_annotations_in_cell = {}
                    max_count = 0
                    for grid_x in range(grid_shape[0]):
                        x_low = lower_bound[0] + grid_x * chunk_size[0]
                        x_high = lower_bound[0] + (grid_x + 1) * chunk_size[0]
                        for grid_y in range(grid_shape[1]):
                            y_low = lower_bound[1] + grid_y * chunk_size[1]
                            y_high = lower_bound[1] + (grid_y + 1) * chunk_size[1]
                            for grid_z in range(grid_shape[2]):
                                z_low = lower_bound[2] + grid_z * chunk_size[2]
                                z_high = lower_bound[2] + (grid_z + 1) * chunk_size[2]
                                remaining_annotations_in_cell[(grid_x, grid_y, grid_z)] = \
                                    (remaining_annotations[:, 0] >= x_low) & (remaining_annotations[:, 0] < x_high) & \
                                    (remaining_annotations[:, 1] >= y_low) & (remaining_annotations[:, 1] < y_high) & \
                                    (remaining_annotations[:, 2] >= z_low) & (remaining_annotations[:, 2] < z_high)
                                max_count = max(max_count,
                                                remaining_annotations_in_cell[(grid_x, grid_y, grid_z)].sum())

                    remaining_annotations_mask = np.zeros(shape=(remaining_annotations.shape[0],), dtype=np.bool_)
                    sample_prop = min(1.0, limit * 1.0 / max_count)
                    for grid_x in range(grid_shape[0]):
                        for grid_y in range(grid_shape[1]):
                            for grid_z in range(grid_shape[2]):
                                rsample = np.random.random_sample((remaining_annotations.shape[0],))
                                emitted = (rsample < sample_prop) & remaining_annotations_in_cell[
                                    (grid_x, grid_y, grid_z)]
                                remaining_annotations_mask |= (rsample >= sample_prop) & \
                                                              remaining_annotations_in_cell[(grid_x, grid_y, grid_z)]
                                # if emitted.sum() == 0:
                                #     continue
                                # write
                                fn = os.path.join(level_cell_annotation_out_folder, f'{grid_x}_{grid_y}_{grid_z}')
                                emitted_annotations = remaining_annotations[emitted, :]
                                with open(fn, 'wb') as outfile:
                                    total_count = emitted_annotations.shape[0]
                                    buf = struct.pack('<Q', total_count)
                                    for (x, y, z, id) in emitted_annotations:
                                        pt_buf = struct.pack('<3f', x, y, z)
                                        buf += pt_buf
                                    # write the ids at the end of the buffer
                                    for (x, y, z, id) in emitted_annotations:
                                        id_buf = struct.pack('<Q', int(id))
                                        buf += id_buf
                                    outfile.write(buf)

                    remaining_annotations = remaining_annotations[remaining_annotations_mask, :]
                    print(f'level {level}: {remaining_annotations_mask.sum()}')
                    if max_count <= limit:
                        assert remaining_annotations.size == 0, remaining_annotations.shape

                    level_info = {
                        "chunk_size": chunk_size,
                        "grid_shape": grid_shape,
                        "key": key,
                        "limit": int(min(limit, max_count))
                    }
                    cell_annotation_info['spatial'].append(level_info)
                    level += 1
                    assert level_resolution[0] == level_resolution[1], level_resolution
                    if level_resolution[0] * 2 < level_resolution[2]:
                        level_resolution = [level_resolution[0] * 2, level_resolution[1] * 2, level_resolution[2]]
                        chunk_size = [chunk_size[0] / 2., chunk_size[1] / 2., chunk_size[2]]
                        grid_shape = [grid_shape[0] * 2, grid_shape[1] * 2, grid_shape[2]]
                    else:
                        level_resolution = [level_resolution[0] * 2, level_resolution[1] * 2, level_resolution[2] * 2]
                        chunk_size = [chunk_size[0] / 2., chunk_size[1] / 2., chunk_size[2] / 2.]
                        grid_shape = [grid_shape[0] * 2, grid_shape[1] * 2, grid_shape[2] * 2]

                with open(os.path.join(cell_annotation_out_folder, 'info'), 'w') as outfile:
                    json.dump(cell_annotation_info, outfile)

    with open(os.path.join(task['out_folder'], 'base.json'), 'w') as outfile:
        print(base_info)
        json.dump(base_info, outfile)


def _callback(result):
    logger.info(f'finished {result}')


def _error_callback(e: BaseException):
    traceback.print_exception(type(e), e, e.__traceback__)
    raise e


if __name__ == "__main__":
    target_folder = os.path.join(io.fs3017_dir(), 'eeum', 'website', 'static', 'web', 'neuroglancer_data')
    target_folder = os.path.join(io.fs3017_data_dir(), 'neuroglancer_data')
    target_folder = os.path.join('/raid', 'neuroglancer_data')
    lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')

    # Previous runs
    if False:
        tsk = {}
        tsk['folder'] = os.path.join(lemur_folder, 'Hotsauce_334A', '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
        tsk['out_folder_name'] = '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN'
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(tsk['folder'], 'hj_aligned', 'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
        tsk['channel_names'] = ['Brightfield', 'DAPI', 'SMI99', 'VGluT2', 'NeuN']
        tsk['resolution'] = [652.761, 652.761, 100000.]
        tsk['bigregion_annotation_name'] = os.path.join(tsk['folder'], 'interns_edited_results',
                                                        'edited_merge_20201001_1.reganno')
        tsk['bigregion_annotation_mesh_folder'] = os.path.join(tsk['folder'], 'bigregion_meshes_v2')

        target_folder = os.path.join('/Volumes', 'T7Touch', 'neuroglancer_data')
        lemur_folder = os.path.join('/Volumes', 'T7Touch', 'HS2')
        tsk = {'folder': lemur_folder, 'out_folder_name': '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN'}
        tsk['img_scale'] = 1. / 16
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(os.path.expanduser('~'), 'Documents', 'jinnylab-annotation-v3_1',
                                       'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
        tsk['channel_names'] = ['Brightfield', 'DAPI', 'SMI99', 'VGluT2', 'NeuN']
        tsk['resolution'] = [652.761, 652.761, 100000.]
        tsk['bigregion_annotation_name'] = os.path.join(tsk['folder'], 'interns_edited_results',
                                                        'edited_merge_20201001_1.reganno')
        tsk['bigregion_annotation_mesh_folder'] = os.path.join(tsk['folder'], 'bigregion_meshes_v2')

        tsk = {}
        tsk['folder'] = os.path.join(lemur_folder, 'Hotsauce_334A', '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
        tsk['out_folder_name'] = '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN'
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(tsk['folder'], 'hj_aligned', 'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
        tsk['channel_names'] = ['Brightfield', 'DAPI', 'SMI99', 'VGluT2', 'NeuN']
        tsk['resolution'] = [652.761, 652.761, 100000.]
        tsk['bigregion_annotation_name'] = os.path.join(tsk['folder'], 'interns_edited_results',
                                                        'edited_merge_20201001_1.reganno')
        tsk['bigregion_annotation_mesh_folder'] = os.path.join(tsk['folder'], 'bigregion_meshes_v2')

        target_folder = os.path.join('/Volumes', 'T7Touch', 'neuroglancer_data')
        lemur_folder = os.path.join('/Volumes', 'T7Touch', 'HS2')
        tsk = {'folder': lemur_folder, 'out_folder_name': '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN'}
        tsk['img_scale'] = 1. / 16
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(os.path.expanduser('~'), 'Documents', 'jinnylab-annotation-v3_1',
                                       'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
        tsk['channel_names'] = ['Brightfield', 'DAPI', 'SMI99', 'VGluT2', 'NeuN']
        tsk['resolution'] = [652.761, 652.761, 100000.]
        tsk['bigregion_annotation_name'] = os.path.join(tsk['folder'], 'subregion',
                                                        'combined.reganno')
        # tsk['bigregion_annotation_mesh_folder'] = os.path.join(tsk['folder'], 'bigregion_meshes_v2')

        target_folder = os.path.join('/Volumes', 'T7Touch', 'neuroglancer_data')
        lemur_folder = os.path.join('/Volumes', 'T7Touch', 'HS2')
        tsk = {'folder': lemur_folder, 'out_folder_name': '20190813_jellybean_FOXP2_SMI32_NeuN'}
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(os.path.expanduser('~'), 'Documents', '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN',
                                       'normalize_DAPI', 'Lemur-H_SMI99_VGluT2_NeuN_all_background_corrected_4.nim')
        tsk['channel_names'] = ['DAPI', ]
        tsk['resolution'] = [10444.2, 10444.2, 100000.]
        # tsk['bigregion_annotation_name'] = os.path.join(tsk['folder'], 'subregion',
        #                                                 'combined.reganno')
        # tsk['bigregion_annotation_mesh_folder'] = os.path.join(tsk['folder'], 'bigregion_meshes_v2')

        target_folder = os.path.join('/Volumes', 'T7Touch', 'neuroglancer_data')
        lemur_folder = os.path.join('/Volumes', 'T7Touch', 'HS2')
        tsk = {'folder': lemur_folder, 'out_folder_name': '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN'}
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(os.path.expanduser('~'), 'Documents', '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN',
                                       'normalize_DAPI', 'Lemur-H_SMI99_VGluT2_NeuN_all_background_corrected.nim')
        tsk['channel_names'] = ['DAPI', ]
        tsk['resolution'] = [10444.2, 10444.2, 100000.]

        target_folder = os.path.join('/Volumes', 'T7Touch', 'neuroglancer_data')
        lemur_folder = os.path.join('/Volumes', 'T7Touch', 'HS2')
        tsk = {'folder': lemur_folder, 'out_folder_name': '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN'}
        tsk['img_scale'] = 1. / 16
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(os.path.expanduser('~'), 'Documents', 'jinnylab-annotation-v3_1',
                                       'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
        tsk['channel_names'] = ['Brightfield', 'DAPI', 'SMI99', 'VGluT2', 'NeuN']
        tsk['resolution'] = [652.761, 652.761, 100000.]
        tsk['bigregion_annotation_name'] = os.path.join(tsk['folder'], 'interns_edited_results',
                                                        'edited_merge_20201001_1.reganno')
        tsk['bigregion_annotation_mesh_folder'] = os.path.join(tsk['folder'], 'bigregion_meshes_v2')
        tsk['cell_annotations'] = {'DAPI_detection': os.path.join(os.path.expanduser('~'), 'Documents',
                                                                  '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN',
                                                                  'cell_detection',
                                                                  'combined',
                                                                  'Lemur-H_SMI99_VGluT2_NeuN_ch1_detection.npy'),
                                   'NeuN_detection': os.path.join(os.path.expanduser('~'), 'Documents',
                                                                  '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN',
                                                                  'cell_detection',
                                                                  'combined',
                                                                  'Lemur-H_SMI99_VGluT2_NeuN_ch4_detection.npy'),
                                   }

    # For mesh
    if False:
        target_folder = os.path.join('/raid/eeum_website/static', 'neuroglancer_data')
        lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')
        annotation_folder  = os.path.join(io.jinny_nas_dir(), 'Mouse_Lemur', '8_processed_results')
        tsk = {}
        tsk = {'folder': lemur_folder, 'out_folder_name': 'Atlas'}
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['annotation_name'] = os.path.join(annotation_folder, '0_final_annotation', 'eLemur_annotation_v2.reganno')
        tsk['annotation_tab_name'] = 'Atlas'
        tsk['annotation_exclude_regions'] = [1979, 9101, 9102, 9103]  # # Indusium Griseum, ventricles
        tsk['annotation_hide_regions'] = [3155, 3156, 3157, 3158, 3159]   # whole layers, hide them to show subregions
        tsk['annotation_resolution'] = [10000., 10000., 100000.]
        # not necessary if we have image
        tsk['annotation_depth'] = 180
        tsk['annotation_height'] = 1300
        tsk['annotation_width'] = 1800
        #
        tsk['annotation_mesh_folder'] = os.path.join(annotation_folder, '0_final_mesh',  'ver-2_1')
        tsk['annotation_mesh_resolution'] = [10000., 10000., 10000.]


    # For reference data
    if False:
        target_folder = os.path.join('/raid/eeum_website/static', 'neuroglancer_data')
        lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')
        annotation_folder  = os.path.join(io.jinny_nas_dir(), 'Mouse_Lemur', '8_processed_results')
        tsk = {}
        tsk['folder'] = os.path.join(lemur_folder, 'Hotsauce_334A', '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
        tsk['out_folder_name'] = '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN'
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(tsk['folder'], 'background_corrected', 'aligned', 'corrected', 'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
        # tsk['img_name'] = os.path.join(io.jinny_nas_dir(), 'HyungJu', 'lemur', 'Hotsauce_SMI99_VGluT2_NeuN', 'Hotsauce_SMI99_VGluT2_NeuN-manual_correction.nim')
        # tsk['img_name'] = os.path.join(annotation_folder, 'Hotsauce_SMI99_VGluT2_NeuN', 'Hotsauce_SMI99_VGluT2_NeuN-manual_correction.nim')
        tsk['channel_names'] = ['Brightfield', 'DAPI', 'SMI99', 'VGluT2', 'NeuN']
        tsk['resolution'] = [652.761, 652.761, 100000.]

        tsk['annotation_name'] = os.path.join(annotation_folder, '0_final_annotation', 'reference_annotation.reganno')
        tsk['annotation_tab_name'] = 'Atlas'
        # tsk['annotation_exclude_regions'] = [1979, 9101, 9102, 9103]  # # Indusium Griseum, ventricles
        # tsk['annotation_hide_regions'] = [3155, 3156, 3157, 3158, 3159]  # whole layers, hide them to show subregions
        tsk['annotation_resolution'] = [652.761, 652.761, 100000.]
        tsk['annotation_scale'] = 1. / 16
        # not necessary if we have image
        # tsk['annotation_depth'] = 180
        # tsk['annotation_height'] = 1300
        # tsk['annotation_width'] = 1800
        # tsk['annotation_exclude_regions'] = [1979, 9101, 9102, 9103]  # # Indusium Griseum, ventricles
        tsk['annotation_hide_regions'] = [3155, 3156, 3157, 3158, 3159]   # whole layers, hide them to show subregions
        #
        # tsk['annotation_mesh_folder'] = os.path.join(annotation_folder, '0_final_mesh',  'ver-2_1')
        # tsk['annotation_mesh_resolution'] = [10000., 10000., 10000.]


    # For non-reference data
    if True:
        target_folder = os.path.join('/raid/eeum_website/static', 'neuroglancer_data')
        lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')
        tsk = {}
        tsk['folder'] = os.path.join(lemur_folder, 'Fig_325AA', '180918_Lemur-Fig_PV_TH_NeuN')
        tsk['out_folder_name'] = 'eLemur-B2'
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(tsk['folder'], 'background_corrected', 'aligned', 'corrected', 'Lemur-F_PV_TH_NeuN_all.nim')
        tsk['channel_names'] = ['Brightfield', 'DAPI', 'PV', 'TH', 'NeuN']
        tsk['resolution'] = [652.761, 652.761, 100000.]

        tsk['annotation_name'] = os.path.join(tsk['folder'], 'matched_eLemur_annotation.reganno')
        tsk['annotation_tab_name'] = 'Atlas'
        # tsk['annotation_exclude_regions'] = [1979, 9101, 9102, 9103]  # # Indusium Griseum, ventricles
        # tsk['annotation_hide_regions'] = [3155, 3156, 3157, 3158, 3159]  # whole layers, hide them to show subregions
        tsk['annotation_resolution'] = [652.761, 652.761, 100000.]
        tsk['annotation_scale'] = 1. / 4
        # not necessary if we have image
        # tsk['annotation_depth'] = 180
        # tsk['annotation_height'] = 1300
        # tsk['annotation_width'] = 1800
        # tsk['annotation_exclude_regions'] = [1979, 9101, 9102, 9103]  # # Indusium Griseum, ventricles
        tsk['annotation_hide_regions'] = [3155, 3156, 3157, 3158, 3159]   # whole layers, hide them to show subregions

        # run task
        tasks = [tsk, ]
        for task in tasks:
            lemur_to_neuroglancer(task)


    # For Mouse data
    if False:
        target_folder = os.path.join('/raid/eeum_website/static', 'neuroglancer_data')
        mouse_folder = os.path.join(io.jinny_nas_dir(), 'Project', 'Project', 'E:I')
        tsk = {}
        tsk['folder'] = os.path.join(mouse_folder, '3M', 'JK979#1')
        tsk['out_folder_name'] = 'JK979_PV_VGluT2_NeuN'
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(tsk['folder'], '20210603_JK979_Set2_aligned_full_res.nim')
        # tsk['img_name'] = os.path.join(io.jinny_nas_dir(), 'HyungJu', 'lemur', 'Hotsauce_SMI99_VGluT2_NeuN', 'Hotsauce_SMI99_VGluT2_NeuN-manual_correction.nim')
        # tsk['img_name'] = os.path.join(annotation_folder, 'Hotsauce_SMI99_VGluT2_NeuN', 'Hotsauce_SMI99_VGluT2_NeuN-manual_correction.nim')
        tsk['channel_names'] = ['DAPI', 'PV', 'VGluT2', 'NeuN']
        tsk['resolution'] = [652.761, 652.761, 100000.]
        # run task
        tasks = [tsk, ]
        for task in tasks:
            lemur_to_neuroglancer(task)


    if False:
        target_folder = os.path.join('/raid/eeum_website/static', 'neuroglancer_data')
        mouse_folder = os.path.join(io.jinny_nas_dir(), 'Project', 'Project', 'E:I')
        tsk = {}
        tsk['folder'] = os.path.join(mouse_folder, '3M', 'JK980#1')
        tsk['out_folder_name'] = 'JK980_PV_TH_NeuN'
        tsk['out_folder'] = os.path.join(target_folder, tsk['out_folder_name'])
        tsk['img_name'] = os.path.join(tsk['folder'], 'JK980_NeuN_TH_PV_aligned.nim')
        # tsk['img_name'] = os.path.join(io.jinny_nas_dir(), 'HyungJu', 'lemur', 'Hotsauce_SMI99_VGluT2_NeuN', 'Hotsauce_SMI99_VGluT2_NeuN-manual_correction.nim')
        # tsk['img_name'] = os.path.join(annotation_folder, 'Hotsauce_SMI99_VGluT2_NeuN', 'Hotsauce_SMI99_VGluT2_NeuN-manual_correction.nim')
        tsk['channel_names'] = ['DAPI', 'PV', 'TH', 'NeuN']
        tsk['resolution'] = [652.761, 652.761, 100000.]
        # run task
        tasks = [tsk, ]
        for task in tasks:
            lemur_to_neuroglancer(task)

    # # run task
    # tasks = [tsk, ]
    # for task in tasks:
    #     lemur_to_neuroglancer(task)

    print('done')
