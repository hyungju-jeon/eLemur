from pathlib import PurePath

from zimg import *
import utils.io as io
import numpy as np
import math
import cv2
import glob
import sys
from utils.logger import setup_logger
from utils import detection
from utils import nim_roi

logger = setup_logger()

"""
macro "Cross-Profiles"{
    print("\\Clear");
    close("* Peak*");
    print("nimage " + ojNImages());
    print("nobject " + ojNObjects());

    for(img = 1; img <= ojNImages(); img++){
        print("image name " + ojGetImageName(img));
        for(obj = ojFirstObject(img); obj <= ojLastObject(img); obj++){
            ojSelectObject(obj);
            print("obj " + obj + " nitems " + ojNItems("NeuCount"));
            minX = 20000;
            minY = 20000;
            maxX = 0;
            maxY = 0;
            for (itm = 1; itm <= ojNItems("NeuCount"); itm++){
                ojSelectItem("*", itm);
                //print("npoints " + ojNPoints());
                //print("" + ojXPos(1) + ", " + ojYPos(1));
                minX = minOf(minX, ojXPos(1));
                minY = minOf(minY, ojYPos(1));
                maxX = maxOf(maxX, ojXPos(1));
                maxY = maxOf(maxY, ojYPos(1));
            }
            print("" + minX +  ", " + minY +  ", " + maxX +  ", " + maxY);
        }
    }
    selectWindow("Log");
}
"""


def count_num_ojbects(label_data: np.ndarray):
    labels = np.unique(label_data)
    labels = labels[labels != 0]
    return labels.shape[0]


def compare_with_manual():
    roi_xyxy = np.array([
        [1719, 3077.75, 1852.75, 3214.75],
        [4174.3335, 1606.6666, 4307.3335, 1738.6666],
        [5542, 1608.3334, 5671, 1743],
        [5541.6665, 2973.3333, 5673.6665, 3104.6665],
        [5545.6665, 4340.3335, 5674, 4472],
        [4175.6665, 4339.6665, 4306.6665, 4473.6665],
        [4175, 5706, 4308.3335, 5838.3335],
        [2810, 5709.6665, 2943.6667, 5839.6665],
        [2803.6667, 7075, 2932, 7203.3335],
        [1442, 7077, 1572, 7209.6665],
        [1443.3334, 5706.6665, 1577.6666, 5843],
        [1445.3333, 4341, 1571.3334, 4474.3335],
        [2935, 4388.6665, 2937.6667, 4450],
        [2807.3333, 2975.3333, 2940, 3103.6667],
        [4175.3335, 2973.6667, 4309.3335, 3040.3333],
        [2811.5, 8440, 2938, 8553],
        [3682.3333, 1883.6666, 3815, 2011],
        [3686, 3242.3333, 3812.6667, 3376],
        [2320.25, 3243, 2451, 3377.75],
        [2316.5, 4605.5, 2449, 4737.5],
        [2943.6667, 2696, 3075.3333, 2826.3333],
        [4311.3335, 2697.3333, 4438.3335, 2827.6667],
        [4307, 4063.3333, 4436.6665, 4186.3335],
        [2943.6667, 4063.6667, 3076.3333, 4185.3335],
        [1578.6666, 4063.3333, 1711, 4189.6665],
        [1580.3334, 5424, 1710, 5556],
        [2943, 5425, 3075.3333, 5555.3335],
        [1581.6666, 6786, 1713.6666, 6915.6665],
        [2944.6667, 6786.6665, 3072.6667, 6916.6665],
        [1584.5, 8148.5, 1712, 8255.5],
        [3526, 2725.3333, 3659, 2856.3333],
        [3527.6667, 4090.3333, 3658, 4219],
        [4887.6665, 2727, 5020.6665, 2857.3333],
        [6250, 2726, 6382, 2858.6667],
        [4887, 4090, 5019.6665, 4218.6665],
        [4889.6665, 5451, 5018.6665, 5581.3335],
        [3527, 5452, 3658, 5580.6665],
        [2164, 5450.3335, 2293, 5582.6665],
        [2162.6667, 4087, 2292.6667, 4216.3335],
        [3529.3333, 6811.6665, 3656.3333, 6946.3335],
        [2161.6667, 6814.3335, 2294.6667, 6947.3335],
        [2167, 8181.6665, 2296.3333, 8307],
        [5163.6665, 1431.3334, 5297.3335, 1565.6666],
        [3805.3333, 1431.3334, 3929.3333, 1557],
        [5166, 2794.3333, 5296.6665, 2922],
        [3805, 2793.6667, 3931, 2925],
        [2440.6667, 2795, 2571.6667, 2926.6667],
        [1077, 4154.5, 1207, 4289.3335],
        [2440, 4155.5, 2573.3333, 4285.6665],
        [3805.6667, 4155.3335, 3935.3333, 4285.3335],
        [3804, 5518.6665, 3931.3333, 5648.3335],
        [2443, 5522.6665, 2575, 5650.6665],
        [1077.3334, 5520.3335, 1209.3334, 5651.6665],
        [1079.6666, 6882.6665, 1210.6666, 7010.6665],
        [2439.6667, 6883, 2572.6667, 7016],
        [2848.3333, 2495, 2892.3333, 2505.6667],
        [4158.6665, 2371.6667, 4290.3335, 2502.6667],
        [2796.3333, 3733, 2927.6667, 3867.3333],
        [4163.6665, 3734, 4288.6665, 3865.6667],
        [5520.6665, 3733.3333, 5653.3335, 3864],
        [6885.6665, 3732.3333, 7018, 3857],
        [5523.3335, 5085.3335, 5657.3335, 5228],
        [4160.3335, 5095.3335, 4285.6665, 5227],
        [2797.1667, 5100, 2919.5, 5229.25],
        [2796, 6456.3335, 2929.6667, 6584],
        [4157.6665, 6461, 4292.6665, 6587.6665],
        [4161.6665, 7818.3335, 4287, 7948.3335],
        [2801, 7819, 2927, 7953.6665],
        [2834, 2539.6667, 2919.6667, 2631],
        [4154, 2500, 4281.6665, 2629],
        [5519, 2516.3333, 5649.6665, 2631.3333],
        [6876.6665, 3861.3333, 7009.3335, 3990.3333],
        [5515.6665, 3864, 5646.6665, 3995.6667],
        [4155.3335, 3860.6667, 4286.6665, 3992.6667],
        [2791.3333, 3862.6667, 2917.6667, 3992.3333],
        [2791.3333, 5226.3335, 2916.3333, 5356.6665],
        [4152.3335, 5227.6665, 4282.3335, 5355],
        [5514.6665, 5224.6665, 5646.6665, 5356],
        [4154, 6595, 4282.3335, 6716.6665],
        [2791.3333, 6592.6665, 2917, 6717],
        [1427.3334, 6591.3335, 1551.3334, 6719.6665],
        [1464.3334, 7953.6665, 1556.3334, 8049],
        [2787.3333, 7949.6665, 2921, 8079],
        [4153, 7949.6665, 4271.6665, 8082.6665],
    ])

    assert roi_xyxy.shape[0] == 84 and roi_xyxy.shape[1] == 4

    det_folder = os.path.join(io.fs3017_dir(), 'eeum', 'lemur', 'detection')
    for roi_index in range(roi_xyxy.shape[0]):
        if roi_index == 0:
            img_slice = 41
        elif roi_index < 16:
            img_slice = 36
        elif roi_index < 20:
            img_slice = 40
        elif roi_index < 30:
            img_slice = 39
        elif roi_index < 42:
            img_slice = 38
        elif roi_index < 55:
            img_slice = 37
        elif roi_index < 68:
            img_slice = 35
        elif roi_index < 84:
            img_slice = 34
        else:
            assert False, roi_index

        det_filename = os.path.join(det_folder, f'Lemur-H_SMI99_VGluT2_NeuN_{img_slice}_s1_ch4_detection.nim')
        # print(det_filename)
        img = ZImg(det_filename)
        label_img = img.data[0][0][0]
        # print(label_img.min(), label_img.max())

        scale = 1.0 / 0.75
        x_start = int(math.floor(roi_xyxy[roi_index, 0] * scale))
        y_start = int(math.floor(roi_xyxy[roi_index, 1] * scale))
        x_end = int(math.ceil(roi_xyxy[roi_index, 2] * scale))
        y_end = int(math.ceil(roi_xyxy[roi_index, 3] * scale))

        print(count_num_ojbects(label_img[y_start:y_end, x_start:x_end]))


def detect_cells(detector, filename: str, channel: int, label_2d_file: str):
    if os.path.exists(label_2d_file):
        logger.info(f'{label_2d_file} finished')
    else:
        infoList = ZImg.readImgInfos(filename)
        assert len(infoList) == 1 and infoList[0].numTimes == 1
        print('image', infoList[0])

        logger.info(f'detecting nuclei from {filename} to {label_2d_file}')

        img = ZImg(filename, ZImgRegion(ZVoxelCoordinate(0, 0, 0, channel, 0),
                                        ZVoxelCoordinate(-1, -1, -1, channel + 1, 1)))
        img = ZImg(filename)

        print(img.data[0].shape)
        data = img.data[0][channel]
        print(data.shape)
        print(data.min())
        print(data.max())

        label_image = np.full(shape=data.shape, fill_value=0, dtype=np.int32)
        rois = [[] for i in range(data.shape[0])]
        spline_rois = [[] for i in range(data.shape[0])]
        for idx, image in enumerate(data):
            print(idx)
            # detections = detector.run_on_opencv_image(image, normalize=False)
            detections = detector.run_on_opencv_image(image, normalize=True, tile_size=512)
            label_image[idx, :, :] = detections['label_image'][0, :, :]  # 0 is the class id of nuclei

            for label in np.unique(detections['label_image'][0, :, :]):
                if label == 0:
                    continue
                mask = label == detections['label_image'][0, :, :]

                polys = nim_roi.mask_to_polygons(mask)
                # assert len(polys) == 1, (im_name, polys)
                if polys:
                    rois[idx].append(polys[0])

                # sample roi to spline curve for possible editing
                splines = nim_roi.mask_to_sampled_splines(mask)
                # assert len(splines) == 1, (im_name, splines)
                if splines:
                    spline_rois[idx].append(splines[0])

        nim_roi.write_polygon_rois(rois,
                                   roi_name=label_2d_file.replace('.nim', '.nimroi'))
        nim_roi.write_spline_rois(spline_rois,
                                  roi_name=label_2d_file.replace('.nim', '_spline.nimroi'))

        label_img = ZImg(label_image[np.newaxis, :, :, :])
        label_img.save(label_2d_file)


def detect_cells_all():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    import models.nuclei.nuclei.predictor as nuclei_detection
    # detector = nuclei_detection.get_detector(use_gpu=True, parallel=False)
    detector = nuclei_detection.get_lemur_cell_detector(use_gpu=True, parallel=False, scale_image_before_inference=2.0)

    folder = os.path.join(io.fs3017_dir(), 'eeum/lemur/Lemur-J_FOXP2_SMI32_NeuN')
    for filename in glob.iglob(os.path.join(folder, '*_Sum.lsm')):
        print(filename)
        for ch in [0, 1]:
            nuclei_label_2d_file = os.path.join(folder, f'{PurePath(filename).name}_ch{ch}_label_2d.nim')
            detect_cells(detector, filename, channel=ch, label_2d_file=nuclei_label_2d_file)
            nuclei_label_file = os.path.join(folder, f'{PurePath(filename).name}_ch{ch}_label.nim')
            detection.merge_2d_label(nuclei_label_2d_file, nuclei_label_file)


if __name__ == "__main__":
    detect_cells_all()

    for img_slice, scene in zip((31, 32), (0, 1)):
        fixed_img = ZImg(
            os.path.join(io.fs3017_dir(), 'eeum/lemur/Jellybean_289BD/20190813_jellybean_FOXP2_SMI32_NeuN',
                         f'Lemur-J_FOXP2_SMI32_NeuN_{img_slice}.czi'),
            region=ZImgRegion(ZVoxelCoordinate(0, 0, 0, 1, 0),
                              ZVoxelCoordinate(-1, -1, -1, 2, -1)),
            scene=scene
        )
        fixed_img_dapi_detection_label_img = ZImg(
            os.path.join(io.fs3017_dir(), 'eeum/lemur/detection',
                         f'Lemur-J_FOXP2_SMI32_NeuN_{img_slice}_s{scene}_ch1_detection.nim')
        )
        fixed_img_dapi_label_data = fixed_img_dapi_detection_label_img.data[0][0][0]
        fixed_img_neun_detection_label_img = ZImg(
            os.path.join(io.fs3017_dir(), 'eeum/lemur/detection',
                         f'Lemur-J_FOXP2_SMI32_NeuN_{img_slice}_s{scene}_ch4_detection.nim')
        )
        fixed_img_neun_label_data = fixed_img_neun_detection_label_img.data[0][0][0]
        for tile in range(1, 16):
            moving_img_filename = os.path.join(io.fs3017_dir(), 'eeum/lemur/Lemur-J_FOXP2_SMI32_NeuN',
                                               f'Lemur-J_FOXP2_SMI32_NeuN_{img_slice}-{scene+1}_Visual-cortex_MTS_L{tile}_Sum.lsm')
            moving_img = ZImg(
                moving_img_filename,
                region=ZImgRegion(ZVoxelCoordinate(0, 0, 0, 0, 0),
                                  ZVoxelCoordinate(-1, -1, -1, 1, -1)),
                scene=0)
            scale = moving_img.info.voxelSizeX / fixed_img.info.voxelSizeX
            moving_data = moving_img.data[0][0].max(axis=0, keepdims=False)
            print(moving_data.shape, scale)
            width = int(moving_data.shape[1] * scale)
            height = int(moving_data.shape[0] * scale)
            moving_data = cv2.resize(moving_data, (width, height), interpolation=cv2.INTER_CUBIC)
            print(moving_data.shape)
            moving_img = ZImg(moving_data[np.newaxis, np.newaxis, :, :])

            ncc_match = ZImgNCCMatch(fixed_img, moving_img)
            moving_loc_res = ncc_match.computeMovingImgOffset()
            print(f'moving image: {PurePath(moving_img_filename).name}')
            print(f'moving image location is {moving_loc_res[0]} with scale {scale}')

            moving_img_dapi_detection_label_img = ZImg(
                os.path.join(io.fs3017_dir(), 'eeum/lemur/Lemur-J_FOXP2_SMI32_NeuN',
                             f'Lemur-J_FOXP2_SMI32_NeuN_{img_slice}-{scene+1}_Visual-cortex_MTS_L{tile}_Sum.lsm_ch0_label.nim')
            )
            moving_img_neun_detection_label_img = ZImg(
                os.path.join(io.fs3017_dir(), 'eeum/lemur/Lemur-J_FOXP2_SMI32_NeuN',
                             f'Lemur-J_FOXP2_SMI32_NeuN_{img_slice}-{scene+1}_Visual-cortex_MTS_L{tile}_Sum.lsm_ch1_label.nim')
            )
            axioscan_dapi = count_num_ojbects(
                fixed_img_dapi_label_data[
                moving_loc_res[0].y:moving_loc_res[0].y + height,
                moving_loc_res[0].x:moving_loc_res[0].x + width
                ])
            confocal_dapi = count_num_ojbects(moving_img_dapi_detection_label_img.data[0][0])
            axioscan_neun = count_num_ojbects(
                fixed_img_neun_label_data[
                moving_loc_res[0].y:moving_loc_res[0].y + height,
                moving_loc_res[0].x:moving_loc_res[0].x + width
                ])
            confocal_neun = count_num_ojbects(moving_img_neun_detection_label_img.data[0][0])
            print('axioscan_dapi, confocal_dapi, axioscan_neun, confocal_neun')
            print(f'{axioscan_dapi}, {confocal_dapi}, {axioscan_neun}, {confocal_neun}')

    print('done')
