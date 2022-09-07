
import os
import sys
sys.path.append('hyungju/code/fenglab')

from pathlib import PurePath
import multiprocessing
import traceback

from zimg import *
import utils.io as io
import utils.detection as detection_utils
from utils.logger import setup_logger

import models.nuclei.nuclei.predictor as nuclei_detection


logger = setup_logger()


def detect_nuclei(detector, filename: str, scene: int, channel: int, label_2d_file: str):
    if os.path.exists(label_2d_file):
        logger.info(f'{label_2d_file} finished')
    else:
        infoList = ZImg.readImgInfos(filename)
        assert infoList[scene].numTimes == 1
        print('image', infoList[scene])

        logger.info(f'detecting nuclei from {filename} to {label_2d_file}')

        img = ZImg(filename, ZImgRegion(ZVoxelCoordinate(0, 0, 0, channel, 0),
                                        ZVoxelCoordinate(-1, -1, -1, channel + 1, 1)),
                   scene=scene)

        data = img.data[0][0]
        print(data.shape)

        label_image = np.full(shape=data.shape, fill_value=0, dtype=np.int32)
        for idx, image in enumerate(data):
            print(idx)
            detections = detector.run_on_opencv_image(image, normalize=False, tile_size=256)
            label_image[idx, :, :] = detections['id_to_label_image'][1]  # 1 is the class id of nuclei

        label_img = ZImg(label_image[np.newaxis, :, :, :])
        label_img.save(label_2d_file)
        detection_utils.label_to_RGB(label_image,
                                     os.path.join(PurePath(label_2d_file).parents[0],
                                                  PurePath(label_2d_file).name + '_rgb.nim'))


def detector_initializer(gpu_queue):
    global detector
    gpu_id = gpu_queue.get()  # block till one gpu_id is available
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # import utils.nuclei_detection as nuclei_detection
    import models.nuclei.nuclei.predictor as nuclei_detection
    detector = nuclei_detection.get_lemur_cell_detector(use_gpu=True, parallel=False, scale_image_before_inference=4.0)


def detector_initializer_PV(gpu_queue):
    global detector
    gpu_id = gpu_queue.get()  # block till one gpu_id is available
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # import utils.nuclei_detection as nuclei_detection
    import models.nuclei.nuclei.predictor as nuclei_detection
    detector = nuclei_detection.get_lemurPV_cell_detector(use_gpu=True, parallel=False, scale_image_before_inference=4.0)


def do_lemur_nuclei_detection_czi(detector, folder, prefix, channels=(1, 4)):
    for img_idx in range(99):
        img_filename = os.path.join(folder, f'{prefix}_{img_idx:02}.czi')
        if not os.path.exists(img_filename):
            logger.info(f'{img_filename} does not exist')
            continue
        num_scenes = len(ZImg.readImgInfos(img_filename))
        for scene in range(num_scenes):
            for ch in channels:
                out_file = os.path.join(folder, f'{prefix}_{img_idx:02}_s{scene}_ch{ch}_detection.nim')

                if os.path.exists(out_file):
                    logger.info(f'nuclei detection {out_file} done')
                else:
                    detect_nuclei(detector, img_filename, scene=scene, channel=ch, label_2d_file=out_file)

                    logger.info(f'nuclei detection {out_file} done')


def do_lemur_nuclei_detection_czi_task(paras: dict):
    folder = paras['folder']
    prefix = paras['prefix']
    channels = paras['channels']
    do_lemur_nuclei_detection_czi(detector, folder, prefix, channels)


def _callback(result):
    logger.info(f'finished {result}')


def _error_callback(e: BaseException):
    traceback.print_exception(type(e), e, e.__traceback__)
    raise e


if __name__ == "__main__":
    num_gpus = 2
    proc_per_gpu = 2
    m = multiprocessing.Manager()
    gpu_queue = m.Queue()
    # initialize the queue with the GPU ids
    for gpu_id in range(num_gpus):
        for _ in range(proc_per_gpu):
            gpu_queue.put(gpu_id)
    logger.info(f'use {gpu_queue.qsize()} gpu processes')


    tasks = [
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                                   '181016_Lemur-Hotsauce_PV_TH_NeuN'),
            'prefix': 'Lemur-H_PV_TH_NeuN',
            'channels': (2,)
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Icecream_225BD',
                                   '190221_icecream_PV_TH_NeuN'),
            'prefix': 'Lemur-I_PV_TH_NeuN',
            'channels': (1,2,4)
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Fig_325AA',
                                   '180918_Lemur-Fig_PV_TH_NeuN'),
            'prefix': 'Lemur-F_PV_TH_NeuN',
            'channels': (2,)
        },
    ]
    with multiprocessing.Pool(processes=gpu_queue.qsize(),
                              initializer=detector_initializer,
                              initargs=(gpu_queue,)) as pool:
        pool.map_async(do_lemur_nuclei_detection_czi_task, tasks,
                       chunksize=1, callback=None, error_callback=_error_callback).wait()

    print('done')
    sys.exit(1)

    tasks = [
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                                   '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN'),
            'prefix': 'Lemur-H_SMI99_VGluT2_NeuN',
            'channels': (1, 4)
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                                   '181016_Lemur-Hotsauce_PV_TH_NeuN'),
            'prefix': 'Lemur-H_PV_TH_NeuN',
            'channels': (1, 4)
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Jellybean_289BD',
                                   '20190813_jellybean_FOXP2_SMI32_NeuN'),
            'prefix': 'Lemur-J_FOXP2_SMI32_NeuN',
            'channels': (1, 4)
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Jellybean_289BD',
                                   '20190827_jellybean_vGluT2_SMI32_vGluT1'),
            'prefix': 'Lemur-J_vGluT2_SMI32_vGluT1',
            'channels': (1, )
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Icecream_225BD',
                                   '190221_icecream_PV_TH_NeuN'),
            'prefix': 'Lemur-I_PV_TH_NeuN',
            'channels': (1, 4)
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Icecream_225BD',
                                   '20190218_icecream_SMI99_NeuN_VGlut2'),
            'prefix': 'Lemur-I_SMI99_VGluT2_NeuN',
            'channels': (1, 4)
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Garlic_320CA',
                                   '181023_Lemur-Garlic_SMI99_VGluT2_M2'),
            'prefix': 'Lemur-G_SMI99_VGluT2_M2',
            'channels': (1, )
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Fig_325AA',
                                   '180914_fig_SMI99_NeuN_VGlut2'),
            'prefix': 'Lemur-F_SMI99_NeuN_VGlut2',
            'channels': (1, 4)
        },
        {
            'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Fig_325AA',
                                   '180918_Lemur-Fig_PV_TH_NeuN'),
            'prefix': 'Lemur-F_PV_TH_NeuN',
            'channels': (1, 4)
        },
    ]
    with multiprocessing.Pool(processes=gpu_queue.qsize(),
                              initializer=detector_initializer,
                              initargs=(gpu_queue,)) as pool:
        pool.map_async(do_lemur_nuclei_detection_czi_task, tasks,
                       chunksize=1, callback=None, error_callback=_error_callback).wait()

    print('done')
    sys.exit(1)

    # mp.set_start_method("spawn", force=True)
    detector = nuclei_detection.get_lemur_cell_detector(use_gpu=True, parallel=False, scale_image_before_inference=4.0)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                          '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
    prefix = 'Lemur-H_SMI99_VGluT2_NeuN'
    do_lemur_nuclei_detection_czi(detector, folder, prefix)
    folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                          '181016_Lemur-Hotsauce_PV_TH_NeuN')
    prefix = 'Lemur-H_PV_TH_NeuN'
    do_lemur_nuclei_detection_czi(detector, folder, prefix)
    folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Jellybean_289BD',
                          '20190813_jellybean_FOXP2_SMI32_NeuN')
    prefix = 'Lemur-J_FOXP2_SMI32_NeuN'
    do_lemur_nuclei_detection_czi(detector, folder, prefix)
    folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Jellybean_289BD',
                          '20190827_jellybean_vGluT2_SMI32_vGluT1')
    prefix = 'Lemur-J_vGluT2_SMI32_vGluT1'
    do_lemur_nuclei_detection_czi(detector, folder, prefix, channels=(1,))

    print('done')
    sys.exit(1)

    folder = os.path.join(io.fs3017_data_dir(), 'lemur/Jellybean_289BD/20190813_jellybean_FOXP2_SMI32_NeuN/')
    out_folder = os.path.join(io.fs3017_dir(), 'eeum', 'lemur', 'detection')
    for slice in (32,):
        filename = os.path.join(folder, f'Lemur-J_FOXP2_SMI32_NeuN_{slice}.czi')
        scene = 1
        for ch in (1, 4):
            out_file = os.path.join(out_folder, f'Lemur-J_FOXP2_SMI32_NeuN_{slice}_s{scene}_ch{ch}_detection.nim')
            detect_nuclei(detector, filename, scene=scene, channel=ch, label_2d_file=out_file)
    sys.exit(0)

    folder = os.path.join(io.jinny_nas_dir(), 'Mouse_Lemur/axioscan/Jellybean_289BD/20190813_jellybean_FOXP2_SMI32_NeuN/')
    out_folder = os.path.join(io.fs3017_dir(), 'eeum', 'lemur', 'detection')
    for slice in (31,):
        filename = os.path.join(folder, f'Lemur-J_FOXP2_SMI32_NeuN_{slice}.czi')
        scene = 0
        for ch in (1, 4):
            out_file = os.path.join(out_folder, f'Lemur-J_FOXP2_SMI32_NeuN_{slice}_s{scene}_ch{ch}_detection.nim')
            detect_nuclei(detector, filename, scene=scene, channel=ch, label_2d_file=out_file)

    folder = os.path.join(io.jinny_nas_dir(), 'Mouse_Lemur/axioscan/Hotsauce_334A/181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN/')
    for slice in range(34, 42):
        filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_{slice}.czi')
        scene = 1
        for ch in (1, 4):
            out_file = os.path.join(out_folder, f'Lemur-H_SMI99_VGluT2_NeuN_{slice}_s{scene}_ch{ch}_detection.nim')
            detect_nuclei(detector, filename, scene=scene, channel=ch, label_2d_file=out_file)

    if True:
        tasks = [
            {
                'folder': os.path.join('/data', 'lemur', '181016_Lemur-Hotsauce_PV_TH_NeuN'),
                'prefix': 'Lemur-H_PV_TH_NeuN',
                'channels': (2,)
            },
        ]
        with multiprocessing.Pool(processes=gpu_queue.qsize(),
                                  initializer=detector_initializer_PV,
                                  initargs=(gpu_queue,)) as pool:
            pool.map_async(do_lemur_nuclei_detection_czi_task, tasks,
                           chunksize=1, callback=None, error_callback=_error_callback).wait()

        print('done')

    if False:
        tasks = [
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                                       '181016_Lemur-Hotsauce_PV_TH_NeuN'),
                'prefix': 'Lemur-H_PV_TH_NeuN',
                'channels': (2,)
            },
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Icecream_225BD',
                                       '190221_icecream_PV_TH_NeuN'),
                'prefix': 'Lemur-I_PV_TH_NeuN',
                'channels': (2,)
            },
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Fig_325AA',
                                       '180918_Lemur-Fig_PV_TH_NeuN'),
                'prefix': 'Lemur-F_PV_TH_NeuN',
                'channels': (2,)
            },
        ]
        with multiprocessing.Pool(processes=gpu_queue.qsize(),
                                  initializer=detector_initializer,
                                  initargs=(gpu_queue,)) as pool:
            pool.map_async(do_lemur_nuclei_detection_czi_task, tasks,
                           chunksize=1, callback=None, error_callback=_error_callback).wait()

        print('done')

    if False:
        tasks = [
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                                       '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN'),
                'prefix': 'Lemur-H_SMI99_VGluT2_NeuN',
                'channels': (1, 4)
            },
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                                       '181016_Lemur-Hotsauce_PV_TH_NeuN'),
                'prefix': 'Lemur-H_PV_TH_NeuN',
                'channels': (1, 4)
            },
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Jellybean_289BD',
                                       '20190813_jellybean_FOXP2_SMI32_NeuN'),
                'prefix': 'Lemur-J_FOXP2_SMI32_NeuN',
                'channels': (1, 4)
            },
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Jellybean_289BD',
                                       '20190827_jellybean_vGluT2_SMI32_vGluT1'),
                'prefix': 'Lemur-J_vGluT2_SMI32_vGluT1',
                'channels': (1, )
            },
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Icecream_225BD',
                                       '190221_icecream_PV_TH_NeuN'),
                'prefix': 'Lemur-I_PV_TH_NeuN',
                'channels': (1, 4)
            },
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Icecream_225BD',
                                       '20190218_icecream_SMI99_NeuN_VGlut2'),
                'prefix': 'Lemur-I_SMI99_VGluT2_NeuN',
                'channels': (1, 4)
            },
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Garlic_320CA',
                                       '181023_Lemur-Garlic_SMI99_VGluT2_M2'),
                'prefix': 'Lemur-G_SMI99_VGluT2_M2',
                'channels': (1, )
            },
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Fig_325AA',
                                       '180914_fig_SMI99_NeuN_VGlut2'),
                'prefix': 'Lemur-F_SMI99_NeuN_VGlut2',
                'channels': (1, 4)
            },
            {
                'folder': os.path.join(io.fs3017_data_dir(), 'lemur', 'Fig_325AA',
                                       '180918_Lemur-Fig_PV_TH_NeuN'),
                'prefix': 'Lemur-F_PV_TH_NeuN',
                'channels': (1, 4)
            },
        ]
        with multiprocessing.Pool(processes=gpu_queue.qsize(),
                                  initializer=detector_initializer,
                                  initargs=(gpu_queue,)) as pool:
            pool.map_async(do_lemur_nuclei_detection_czi_task, tasks,
                           chunksize=1, callback=None, error_callback=_error_callback).wait()

        print('done')

    if False:
        # mp.set_start_method("spawn", force=True)
        detector = nuclei_detection.get_lemur_cell_detector(use_gpu=True, parallel=False, scale_image_before_inference=4.0)
        # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

        folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                              '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
        prefix = 'Lemur-H_SMI99_VGluT2_NeuN'
        do_lemur_nuclei_detection_czi(detector, folder, prefix)
        folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                              '181016_Lemur-Hotsauce_PV_TH_NeuN')
        prefix = 'Lemur-H_PV_TH_NeuN'
        do_lemur_nuclei_detection_czi(detector, folder, prefix)
        folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Jellybean_289BD',
                              '20190813_jellybean_FOXP2_SMI32_NeuN')
        prefix = 'Lemur-J_FOXP2_SMI32_NeuN'
        do_lemur_nuclei_detection_czi(detector, folder, prefix)
        folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Jellybean_289BD',
                              '20190827_jellybean_vGluT2_SMI32_vGluT1')
        prefix = 'Lemur-J_vGluT2_SMI32_vGluT1'
        do_lemur_nuclei_detection_czi(detector, folder, prefix, channels=(1,))

        print('done')

    if False:
        folder = os.path.join(io.fs3017_data_dir(), 'lemur/Jellybean_289BD/20190813_jellybean_FOXP2_SMI32_NeuN/')
        out_folder = os.path.join(io.fs3017_dir(), 'eeum', 'lemur', 'detection')
        for slice in (32,):
            filename = os.path.join(folder, f'Lemur-J_FOXP2_SMI32_NeuN_{slice}.czi')
            scene = 1
            for ch in (1, 4):
                out_file = os.path.join(out_folder, f'Lemur-J_FOXP2_SMI32_NeuN_{slice}_s{scene}_ch{ch}_detection.nim')
                detect_nuclei(detector, filename, scene=scene, channel=ch, label_2d_file=out_file)

    if False:
        folder = os.path.join(io.jinny_nas_dir(), 'Mouse_Lemur/axioscan/Jellybean_289BD/20190813_jellybean_FOXP2_SMI32_NeuN/')
        out_folder = os.path.join(io.fs3017_dir(), 'eeum', 'lemur', 'detection')
        for slice in (31,):
            filename = os.path.join(folder, f'Lemur-J_FOXP2_SMI32_NeuN_{slice}.czi')
            scene = 0
            for ch in (1, 4):
                out_file = os.path.join(out_folder, f'Lemur-J_FOXP2_SMI32_NeuN_{slice}_s{scene}_ch{ch}_detection.nim')
                detect_nuclei(detector, filename, scene=scene, channel=ch, label_2d_file=out_file)

        folder = os.path.join(io.jinny_nas_dir(), 'Mouse_Lemur/axioscan/Hotsauce_334A/181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN/')
        for slice in range(34, 42):
            filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_{slice}.czi')
            scene = 1
            for ch in (1, 4):
                out_file = os.path.join(out_folder, f'Lemur-H_SMI99_VGluT2_NeuN_{slice}_s{scene}_ch{ch}_detection.nim')
                detect_nuclei(detector, filename, scene=scene, channel=ch, label_2d_file=out_file)

