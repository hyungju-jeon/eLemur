import json
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    oapi = OntologiesApi()
    structure_graph = oapi.get_structures_with_sets([1])  # 1 is the id of the adult mouse structure graph
    structure_graph = StructureTree.clean_structures(structure_graph)
    tree = StructureTree(structure_graph)

    mouse_folder = '/Users/hyungju/Desktop/hyungju/Data/mouse_cell'
    result_folder = os.path.join(mouse_folder, 'aligned_detection_cell')

    neun_filename = os.path.join(result_folder, '00_cell_3_count.csv')
    region_filename = os.path.join(result_folder, '00_region_size.csv')

    cell_df = pd.read_csv(neun_filename).T
    region_df = pd.read_csv(region_filename).T

    cell_df = cell_df.rename(columns=cell_df.iloc[0]).drop(cell_df.index[0])
    region_df = region_df.rename(columns=region_df.iloc[0]).drop(region_df.index[0])

    sum_cell = np.sum(cell_df) * 2 * 1.864
    sum_region = np.sum(region_df) * 25 * 25 * 100 * 1e-9

    sample_density = sum_cell / sum_region
    sample_density = sample_density.fillna(0)
    sample_density = sample_density.drop(sample_density[sample_density==0].index)
    sample_density = sample_density.drop(sample_density[sum_region<1].index)

    xls_file = pd.ExcelFile(os.path.join(mouse_folder, 'Rodarie_et_al', 'literature_density.xlsx'), engine='openpyxl')
    sheet_name = "Densities BBCAv2"
    ref_cell = pd.read_excel(xls_file, sheet_name=sheet_name)
    ref_cell = ref_cell.set_index('Brain region')
    ref_density_v2 = ref_cell['Neuron [mm^-3]']
    ref_region_v2 = ref_cell['Volumes [mm^3]']

    ref_cell = pd.read_csv(os.path.join(mouse_folder, 'Ero_et_al', 'Data_Sheet_2_A Cell Atlas for the Mouse Brain.CSV'))
    ref_cell = ref_cell.set_index('Regions')
    ref_density_v1 = ref_cell['Neurons [mm-3]']

    sample_region = sample_density.index
    ref_region_v2 = ref_density_v2.index
    ref_region_v1 = ref_density_v1.index

    common_region = list(set(sample_region) & set(ref_region_v2) & set(ref_region_v1))
    temp = tree.get_structures_by_name(common_region)

    plt.scatter(ref_density_v2[common_region], sample_density[common_region], c=np.array([x['rgb_triplet'] for x in temp])/255)
    plt.gca().set_aspect('equal')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim((1e3, 5e6))
    plt.ylim((1e3, 5e6))
    plt.xlabel('Literature Density')
    plt.ylabel('Detected Density')

    avg_dev = 2.5
    plt.plot([1e3, 5e6], [1e3, 5e6], 'k-', alpha=0.75, zorder=0)
    plt.plot([1e3, 5e6], [avg_dev*1e3, avg_dev*5e6], 'k--', alpha=0.25, zorder=0)
    plt.plot([1e3, 5e6], [1e3/avg_dev, 5e6/avg_dev], 'k--', alpha=0.25, zorder=0)
    plt.show()

    #
    # region_list = [x for x in ra_dict['Regions'].keys() if x > 0]
    # sum_cell_df = pd.DataFrame(columns=region_list)
    # sum_region_df = pd.DataFrame(columns=region_list)
    #
    # cell_df = cell_df.rename(columns=cell_df.iloc[0]).drop(cell_df.index[0])
    # region_df = region_df.rename(columns=region_df.iloc[0]).drop(region_df.index[0])