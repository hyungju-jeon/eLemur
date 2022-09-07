from pathlib import PurePath
import multiprocessing
import traceback

from zimg import *
import utils.io as io
from utils import img_util
from utils import nim_roi
from utils.logger import setup_logger
from utils import region_annotation
from utils import shading_correction


logger = setup_logger()


def detector_initializer(gpu_queue):
    global detector1
    gpu_id = gpu_queue.get()  # block till one gpu_id is available
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    from models.nuclei.nuclei.predictor import get_lemur_bigregion_detector_v6
    detector1 = get_lemur_bigregion_detector_v6(use_gpu=True, parallel=False)


def do_lemur_5ch_bigregion_detection_czi(detector, folder, prefix):
    for img_idx in range(66):
        img_filename = os.path.join(folder, f'{prefix}_{img_idx:02}.czi')
        if not os.path.exists(img_filename):
            logger.info(f'{img_filename} does not exist')
            continue
        num_scenes = len(ZImg.readImgInfos(img_filename))
        ref_ra_filename = os.path.join(io.fs3017_data_dir(), 'lemur', f'Lemur_bigregion_ref.reganno')
        for scene in range(num_scenes):
            ra_filename = os.path.join(folder, f'{prefix}_{img_idx:02}_scene{scene}_bigregion5.reganno')
            if False and os.path.exists(ra_filename):
                logger.info(f'roi to ra {ra_filename} done')
            else:
                read_ratio = 16
                scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
                # img = ZImg(img_filename, region=ZImgRegion(), scene=scene, xRatio=read_ratio, yRatio=read_ratio)
                img_data, img_info = shading_correction.correct_shading(img_filename, scene=scene,
                                                                        inverse_channels=(0,))
                img_data = img_util.imresize(img_data, des_height=img_info.height // read_ratio, des_width=img_info.width // read_ratio)
                logger.info(f'finish reading image from {img_filename}: {img_info}')
                img_data, _ = img_util.normalize_img_data(img_data, min_max_percentile=(2, 98))
                nchs, depth, height, width = img_data.shape

                ra_dict = region_annotation.read_region_annotation(ref_ra_filename)
                for region_id, region_props in ra_dict['Regions'].items():
                    region_props['ROI'] = None

                for slice in range(depth):
                    logger.info(f'slice {slice}')

                    slice_img_data = np.moveaxis(img_data[:, slice, :, :], 0, -1)

                    detections = detector.run_on_opencv_image(slice_img_data, tile_size=20000)
                    for region_id, label_image in detections['id_to_label_image'].items():
                        shapes = nim_roi.label_image_2d_to_spline_shapes(label_image)
                        if len(shapes) > 0:
                            if ra_dict['Regions'][region_id]['ROI'] is None:
                                ra_dict['Regions'][region_id]['ROI'] = {}
                            if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                            if slice not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice] = shapes
                            else:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice].extend(shapes)

                del ra_dict['Regions'][-1]
                ra_dict = region_annotation.transform_region_annotation_dict(ra_dict,
                                                                             lambda coords: coords * read_ratio)
                region_annotation.write_region_annotation_dict(ra_dict, ra_filename)

                logger.info(f'det bigregion {ra_filename} done')


def detector_initializer_2ch(gpu_queue):
    global detector2
    gpu_id = gpu_queue.get()  # block till one gpu_id is available
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    from models.nuclei.nuclei.predictor import get_lemur_bigregion_detector_2ch_v6
    detector2 = get_lemur_bigregion_detector_2ch_v6(use_gpu=True, parallel=False)


def do_lemur_bigregion_detection_czi(detector, folder, prefix):
    for img_idx in range(66):
        img_filename = os.path.join(folder, f'{prefix}_{img_idx:02}.czi')
        if not os.path.exists(img_filename):
            logger.info(f'{img_filename} does not exist')
            continue
        num_scenes = len(ZImg.readImgInfos(img_filename))
        ref_ra_filename = os.path.join(io.fs3017_data_dir(), 'lemur', f'Lemur_bigregion_ref.reganno')
        for scene in range(num_scenes):
            ra_filename = os.path.join(folder, f'{prefix}_{img_idx:02}_scene{scene}_bigregion2.reganno')
            if False and os.path.exists(ra_filename):
                logger.info(f'roi to ra {ra_filename} done')
            else:
                read_ratio = 16
                scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
                # img = ZImg(img_filename, region=ZImgRegion(), scene=scene, xRatio=read_ratio, yRatio=read_ratio)
                img_data, img_info = shading_correction.correct_shading(img_filename, scene=scene,
                                                                        inverse_channels=(0,))
                img_data = img_util.imresize(img_data, des_height=img_info.height // read_ratio, des_width=img_info.width // read_ratio)
                logger.info(f'finish reading image from {img_filename}: {img_info}')
                img_data, _ = img_util.normalize_img_data(img_data, min_max_percentile=(2, 98))
                nchs, depth, height, width = img_data.shape

                ra_dict = region_annotation.read_region_annotation(ref_ra_filename)
                for region_id, region_props in ra_dict['Regions'].items():
                    region_props['ROI'] = None

                for slice in range(depth):
                    logger.info(f'slice {slice}')

                    slice_img_data = np.moveaxis(img_data[0:2, slice, :, :], 0, -1)

                    detections = detector.run_on_opencv_image(slice_img_data, tile_size=20000)
                    for region_id, label_image in detections['id_to_label_image'].items():
                        shapes = nim_roi.label_image_2d_to_spline_shapes(label_image)
                        if len(shapes) > 0:
                            if ra_dict['Regions'][region_id]['ROI'] is None:
                                ra_dict['Regions'][region_id]['ROI'] = {}
                            if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                            if slice not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice] = shapes
                            else:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice].extend(shapes)

                del ra_dict['Regions'][-1]
                ra_dict = region_annotation.transform_region_annotation_dict(ra_dict,
                                                                             lambda coords: coords * read_ratio)
                region_annotation.write_region_annotation_dict(ra_dict, ra_filename)

                logger.info(f'det bigregion {ra_filename} done')


def do_lemur_5ch_bigregion_detection_czi_task(paras: dict):
    folder = paras['folder']
    prefix = paras['prefix']
    do_lemur_5ch_bigregion_detection_czi(detector1, folder, prefix)


def do_lemur_bigregion_detection_czi_task(paras: dict):
    folder = paras['folder']
    prefix = paras['prefix']
    do_lemur_bigregion_detection_czi(detector2, folder, prefix)


def _callback(result):
    logger.info(f'finished {result}')


def _error_callback(e: BaseException):
    traceback.print_exception(type(e), e, e.__traceback__)
    raise e


if __name__ == "__main__":
    num_gpus = 4
    proc_per_gpu = 1
    m = multiprocessing.Manager()
    gpu_queue = m.Queue()
    # initialize the queue with the GPU ids
    for gpu_id in range(num_gpus):
        for _ in range(proc_per_gpu):
            gpu_queue.put(gpu_id)
    logger.info(f'use {gpu_queue.qsize()} gpu processes')

    tasks = [
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Icecream_225BD',
                                   '20190218_icecream_SMI99_NeuN_VGlut2'),
            'prefix': 'Lemur-I_SMI99_VGluT2_NeuN',
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Fig_325AA',
                                   '180914_fig_SMI99_NeuN_VGlut2'),
            'prefix': 'Lemur-F_SMI99_NeuN_VGlut2',
        },
    ]
    with multiprocessing.Pool(processes=gpu_queue.qsize(),
                              initializer=detector_initializer,
                              initargs=(gpu_queue,)) as pool:
        pool.map_async(do_lemur_5ch_bigregion_detection_czi_task, tasks,
                       chunksize=1, callback=None, error_callback=_error_callback).wait()

    print('done 1')
    sys.exit(1)

    tasks = [
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Icecream_225BD',
                                   '20190218_icecream_SMI99_NeuN_VGlut2'),
            'prefix': 'Lemur-I_SMI99_VGluT2_NeuN',
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Fig_325AA',
                                   '180914_fig_SMI99_NeuN_VGlut2'),
            'prefix': 'Lemur-F_SMI99_NeuN_VGlut2',
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                                   '181016_Lemur-Hotsauce_PV_TH_NeuN'),
            'prefix': 'Lemur-H_PV_TH_NeuN',
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Jellybean_289BD',
                                   '20190813_jellybean_FOXP2_SMI32_NeuN'),
            'prefix': 'Lemur-J_FOXP2_SMI32_NeuN',
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Jellybean_289BD',
                                   '20190827_jellybean_vGluT2_SMI32_vGluT1'),
            'prefix': 'Lemur-J_vGluT2_SMI32_vGluT1',
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Icecream_225BD',
                                   '190221_icecream_PV_TH_NeuN'),
            'prefix': 'Lemur-I_PV_TH_NeuN',
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Garlic_320CA',
                                   '181023_Lemur-Garlic_SMI99_VGluT2_M2'),
            'prefix': 'Lemur-G_SMI99_VGluT2_M2',
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Fig_325AA',
                                   '180918_Lemur-Fig_PV_TH_NeuN'),
            'prefix': 'Lemur-F_PV_TH_NeuN',
        },
    ]
    with multiprocessing.Pool(processes=gpu_queue.qsize(),
                              initializer=detector_initializer_2ch,
                              initargs=(gpu_queue,)) as pool:
        pool.map_async(do_lemur_bigregion_detection_czi_task, tasks,
                       chunksize=1, callback=None, error_callback=_error_callback).wait()

    print('done 2')
    sys.exit(1)
