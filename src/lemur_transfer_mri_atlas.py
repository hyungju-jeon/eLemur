import os
import traceback
import ants

from zimg import *
from utils import io
from utils.logger import setup_logger


logger = setup_logger()


def _callback(result):
    logger.info(f'finished {result}')


def _error_callback(e: BaseException):
    traceback.print_exception(type(e), e, e.__traceback__)
    raise e


if __name__ == "__main__":
    folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                          '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
    lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')
    folder = os.path.join(lemur_folder, 'Hotsauce_334A', '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
    img_filename = os.path.join(folder, 'hj_aligned', 'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    midline_filename = os.path.join(folder, 'interns_edited_results', 'sh_cut_in_half.reganno')
    
    mri_folder = '/Users/hyungju/Desktop/hyungju/Data/MIRCen-mouselemur-templateatlas'
    result_folder = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/mri-alignment'
    
    # Fixed volume : blockface
    fix_filename = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/shifted_Hotsauce_blockface-outline_grouped_fix_interpolated_mirror_smooth.tiff'
    fix_filename = '/Users/hyungju/Desktop/hyungju/Result/lemur-blockface/mesh-align/final-alignment/alignment-blockface/blockface_outline_aligned_volume_deform_1.tiff'
    # Moving volume : MRI atlas
    mov_filename = os.path.join(mri_folder, 'MouseLemurLabels_V0.01_noCSF.tif')
    
    mriImg = ants.image_read(mov_filename)
    ants.set_spacing(mriImg, (0.091, 0.091, 0.091))
    movImg = mriImg
    #movImg = ants.get_mask(mriImg, low_thresh=1, high_thresh=None, cleanup=0)
    #movImg = ants.morphology(movImg, operation='close', radius=10, mtype='binary', shape='ball')
    #movImg = ants.iMath(movImg, 'FillHoles')
    
    blockImg = ants.image_read(fix_filename)
    ants.set_spacing(blockImg, (0.01, 0.01, 0.1))
    fixImg = blockImg
    #fixImg = ants.get_mask(blockImg, low_thresh=1, high_thresh=None, cleanup=0)
    #fixImg = ants.morphology(fixImg, operation='close', radius=10, mtype='binary', shape='ball')
    #fixImg = ants.iMath(fixImg, 'FillHoles')
    fixImg = ants.resample_image(fixImg, (0.01,0.01,0.1), 0, 0)

    logger.info(f'running registration')
    mytx = ants.registration(fixed=fixImg , moving=movImg, type_of_transform='antsRegistrationSyN[s]',
    #mytx = ants.registration(fixed=fixImg , moving=movImg, type_of_transform='SyNAggro',                         
                              grad_step=0.1, write_composite_transform=True, verbose=True)
    
    result_filename = os.path.join(result_folder, 'registeredAtlas_noCSF_20210216_test.nii.gz')
    #resultImg = ants.resample_image_to_target(mytx['warpedmovout'], blockImg)
    resultImg = ants.apply_transforms(fixed=blockImg, moving=mriImg,interpolator='nearestNeighbor',
                                           transformlist=mytx['fwdtransforms'] )
    resultImg.to_file(result_filename)
    
    