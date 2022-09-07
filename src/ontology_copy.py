import os
import json


def check_ontology(ontology: dict):
    all_ids = []
    roots = ontology['msg']
    print(f'number of roots: {len(roots)}')

    for root in roots:
        nodes_to_be_visited = [root]
        while nodes_to_be_visited:
            node = nodes_to_be_visited.pop()
            all_ids.append(node['id'])
            for child_node in node['children']:
                assert child_node['parent_structure_id'] == node['id'], f"wrong node with id {child_node['id']}"
                print(f"visiting node {child_node['id']} from parent node {node['id']}")
                nodes_to_be_visited.append(child_node)

    print(f'total {len(all_ids)} regions: {all_ids}')


def check_ontology_v2(ontology: dict):
    all_ids = []
    roots = ontology['msg']
    print(f'number of roots: {len(roots)}')

    def check_tree(tree_root: dict):
        all_ids_in_tree = [tree_root['id']]
        if len(tree_root['children']) > 0:
            for child_tree in tree_root['children']:
                assert child_tree['parent_structure_id'] == tree_root['id'], f"wrong node with id {child_tree['id']}"
                print(f"visiting node {child_tree['id']} from parent node {tree_root['id']}")
                all_ids_in_tree.extend(check_tree(child_tree))
        return all_ids_in_tree

    for root in roots:
        all_ids.extend(check_tree(root))

    print(f'total {len(all_ids)} regions: {all_ids}')


def append_children_regions_to_parent_region(parent_region: dict, children_regions: list):
    parent_region['children'].extend(children_regions)
    for child_region in children_regions:
        child_region['parent_structure_id'] = parent_region['id']


def build_lemur_ontology():
    regions = {}

    regions['root'] = {
        "id": 997,
        "atlas_id": -1,
        "ontology_id": 1,
        "acronym": "root",
        "name": "root",
        "color_hex_triplet": "FFFFFF",
        "graph_order": 0,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": "null",
        "children": [],
    }
    regions['grey'] = {
        "id": 8,
        "atlas_id": 0,
        "ontology_id": 1,
        "acronym": "grey",
        "name": "Grey matter",
        "color_hex_triplet": "BFDAE3",
        "graph_order": 1,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['CH'] = {
        "id": 567,
        "atlas_id": 70,
        "ontology_id": 1,
        "acronym": "CH",
        "name": "Cerebrum",
        "color_hex_triplet": "AFF0FF",
        "graph_order": 2,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Cx'] = {
        "id": 688,
        "atlas_id": 85,
        "ontology_id": 1,
        "acronym": "Cx",
        "name": "Cerebral cortex",
        "color_hex_triplet": "B1FEB9",
        "graph_order": 3,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Neocortex'] = {
        "id": 315,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "NCx",
        "name": "Neocortex",
        "color_hex_triplet": "70FE70",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['L1'] = {
        "id": 3159,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "L1",
        "name": "Layer1",
        "color_hex_triplet": "FF99CC",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['L2&3'] = {
        "id": 3158,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "L2&3",
        "name": "Layer2&3",
        "color_hex_triplet": "FF66CC",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['L4'] = {
        "id": 3157,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "L4",
        "name": "Layer4",
        "color_hex_triplet": "FF33CC",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['L5'] = {
        "id": 3156,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "L5",
        "name": "Layer5",
        "color_hex_triplet": "FF00CC",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['L6'] = {
        "id": 3155,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "L6",
        "name": "Layer6",
        "color_hex_triplet": "990099",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    # regions['L6a'] = {
    #     "id": 3154,
    #     "atlas_id": 746,
    #     "ontology_id": 1,
    #     "acronym": "L6a",
    #     "name": "Layer6a",
    #     "color_hex_triplet": "663399",
    #     "graph_order": 5,
    #     "st_level": "null",
    #     "hemisphere_id": 3,
    #     "parent_structure_id": None,
    #     "children": []
    # }
    # regions['L6b'] = {
    #     "id": 3153,
    #     "atlas_id": 746,
    #     "ontology_id": 1,
    #     "acronym": "L6b",
    #     "name": "Layer6b",
    #     "color_hex_triplet": "660099",
    #     "graph_order": 5,
    #     "st_level": "null",
    #     "hemisphere_id": 3,
    #     "parent_structure_id": None,
    # #     "children": []
    # }
    regions['R'] = {
        "id": 3111,
        "atlas_id": 746,
        "ontology_id": 1,
        "acronym": "R",
        "name": "Remaining region",
        "color_hex_triplet": "39B54A",
        "graph_order": 5,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['HiF'] = {
        "id": 1089,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "HiF",
        "name": "Hippocampal formation",
        "color_hex_triplet": "7ECF4C",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Ent'] = {
        "id": 1929,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "Ent",
        "name": "Entorhinal area",
        "color_hex_triplet": "2FB924",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['CAs'] = {
        "id": 1939,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "CAs",
        "name": "CA fields",
        "color_hex_triplet": "7ECF4A",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['DG'] = {
        "id": 1949,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "DG",
        "name": "Dentate gyrus",
        "color_hex_triplet": "7ED04C",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['S'] = {
        "id": 1959,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "S",
        "name": "Subiculum",
        "color_hex_triplet": "4FC144",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['PrS'] = {
        "id": 1969,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "PrS",
        "name": "Presubiculum",
        "color_hex_triplet": "59B946",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['IG'] = {
        "id": 1979,
        "atlas_id": 135,
        "ontology_id": 1,
        "acronym": "IG",
        "name": "Indusium Griseum",
        "color_hex_triplet": "5358FC",
        "graph_order": 415,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['CNu'] = {
        "id": 623,
        "atlas_id": 77,
        "ontology_id": 1,
        "acronym": "CNu",
        "name": "Cerebral nuclei",
        "color_hex_triplet": "96D6FA",
        "graph_order": 520,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Str'] = {
        "id": 477,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "Str",
        "name": "Striatum",
        "color_hex_triplet": "7ECCED",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Cd'] = {
        "id": 4779,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "Cd",
        "name": "Caudate nucleus",
        "color_hex_triplet": "98D6F9",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Pu'] = {
        "id": 4778,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "Pu",
        "name": "Putamen",
        "color_hex_triplet": "98D6F9",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Pal'] = {
        "id": 803,
        "atlas_id": 241,
        "ontology_id": 1,
        "acronym": "Pal",
        "name": "extended Pallidum",
        "color_hex_triplet": "8599CC",
        "graph_order": 557,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['LGP'] = {
        "id": 4777,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "LGP",
        "name": "Lateral Globus Pallidus",
        "color_hex_triplet": "8599CC",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['MGP'] = {
        "id": 4776,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "MGP",
        "name": "Medial Globus Pallidus",
        "color_hex_triplet": "8599CC",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Bs'] = {
        "id": 343,
        "atlas_id": 42,
        "ontology_id": 1,
        "acronym": "Bs",
        "name": "Brainstem",
        "color_hex_triplet": "C97080",
        "graph_order": 588,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['IBr'] = {
        "id": 1129,
        "atlas_id": 140,
        "ontology_id": 1,
        "acronym": "IBr",
        "name": "Interbrain",
        "color_hex_triplet": "FE7180",
        "graph_order": 589,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Th'] = {
        "id": 549,
        "atlas_id": 351,
        "ontology_id": 1,
        "acronym": "Th",
        "name": "Thalamus",
        "color_hex_triplet": "F45B71",
        "graph_order": 590,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Hy'] = {
        "id": 1097,
        "atlas_id": 136,
        "ontology_id": 1,
        "acronym": "Hy",
        "name": "Hypothalamus",
        "color_hex_triplet": "E64339",
        "graph_order": 655,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['STh'] = {
        "id": 4775,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "STh",
        "name": "Subthalamic Nucleus",
        "color_hex_triplet": "E64339",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['MBr'] = {
        "id": 313,
        "atlas_id": 180,
        "ontology_id": 1,
        "acronym": "MBr",
        "name": "Midbrain",
        "color_hex_triplet": "FF65FF",
        "graph_order": 740,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['SNC'] = {
        "id": 4774,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "SNC",
        "name": "Pars Compacta",
        "color_hex_triplet": "FF65FF",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['SNR'] = {
        "id": 4773,
        "atlas_id": 342,
        "ontology_id": 1,
        "acronym": "SNR",
        "name": "Pars Reticulata",
        "color_hex_triplet": "FF65FF",
        "graph_order": 521,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['HBr'] = {
        "id": 1065,
        "atlas_id": 132,
        "ontology_id": 1,
        "acronym": "HBr",
        "name": "Hindbrain",
        "color_hex_triplet": "FF9A87",
        "graph_order": 801,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Pons'] = {
        "id": 771,
        "atlas_id": 237,
        "ontology_id": 1,
        "acronym": "Pons",
        "name": "Pons",
        "color_hex_triplet": "F98575",
        "graph_order": 802,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['MO'] = {
        "id": 354,
        "atlas_id": 185,
        "ontology_id": 1,
        "acronym": "MO",
        "name": "Medulla",
        "color_hex_triplet": "FE9ACC",
        "graph_order": 849,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['Cb'] = {
        "id": 512,
        "atlas_id": 63,
        "ontology_id": 1,
        "acronym": "Cb",
        "name": "Cerebellum",
        "color_hex_triplet": "F0F07F",
        "graph_order": 927,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['fiber tracts'] = {
            "id": 1009,
        "atlas_id": 691,
        "ontology_id": 1,
        "acronym": "fiber tracts",
        "name": "Fiber tracts",
        "color_hex_triplet": "CBCCCC",
        "graph_order": 1013,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['VS'] = {
        "id": 73,
        "atlas_id": 716,
        "ontology_id": 1,
        "acronym": "VS",
        "name": "ventricular systems",
        "color_hex_triplet": "AAAAAA",
        "graph_order": 1199,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['LV'] = {
        "id": 9101,
        "atlas_id": 716,
        "ontology_id": 1,
        "acronym": "LV",
        "name": "lateral ventricle",
        "color_hex_triplet": "046F12",
        "graph_order": 1199,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['3V'] = {
        "id": 9102,
        "atlas_id": 716,
        "ontology_id": 1,
        "acronym": "3V",
        "name": "third ventricle",
        "color_hex_triplet": "97FEE8",
        "graph_order": 1199,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    regions['4V'] = {
        "id": 9103,
        "atlas_id": 716,
        "ontology_id": 1,
        "acronym": "4V",
        "name": "fourth ventricle",
        "color_hex_triplet": "FD79AB",
        "graph_order": 1199,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    # ....

    parent_to_children_map = {
        'root' : ['grey','fiber tracts', 'VS'],
        'grey' : ['CH', 'Bs', 'Cb'],
        'CH' : ['Cx', 'HiF', 'CNu'],
        'Cx' : ['NCx', 'R'],
        'Isocortex' : ['L1', 'L2&3', 'L4', 'L5', 'L6', 'L6a', 'L6b'],
        'HPF' : ['Ent','CAs', 'DG', 'S', 'PrS', 'IG'],
        'CNU' : ['Str', 'Pal'],
        'STR' : ['Cd', 'Put'],
        'PAL' : ['MGP', 'LGP'],
        'BS' : ['IBr', 'MBr', 'HBr'],
        'IB' : ['Th', 'Hy'],
        'HY' : ['STh'],
        'MB' : ['SNC', 'SNR'],
        'HB' : ['Pons', 'MO'],
        'CB' : [],
        'FT' : [],
        'VS' : ['LV','3V','4V']
    }

    for parent, children in parent_to_children_map.items():
        append_children_regions_to_parent_region(regions[parent],
                                                 [regions[child] for child in children])

    res = {
        "success": "true",
        "id": 0,
        "start_row": 0,
        "num_rows": 1,
        "total_rows": 1,
        "msg": [regions['root']]
    }
    return res


if __name__ == "__main__":
    ontology_copy = os.path.join(os.path.expanduser('~'), 'Downloads', 'lemur_atlas_ontology_v3.json')

    with open(ontology_copy, 'w') as outfile:
        json.dump(build_lemur_ontology(), outfile, indent = 4)

    with open(ontology_copy, 'r') as json_file:
        m_ontology = json.load(json_file)
        check_ontology(m_ontology)
        check_ontology_v2(m_ontology)