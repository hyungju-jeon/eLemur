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
        "name": "Basic cell groups and regions",
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
        "color_hex_triplet": "7E605D",
        "graph_order": 2,
        "st_level": "null",
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": []
    }
    # ....

    parent_to_children_map = {
        'root' : ['grey', ],
        'grey' : ['CH', ]
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
    with open(ontology_filename, 'w') as outfile:
        json.dump(build_lemur_ontology(), outfile)

    ontology_filename = os.path.join(os.path.expanduser('~'), 'code', 'atlas', 'src', 'atlas', 'Resources',
                                     'ontology', 'mouse_brain_atlas.json')
    with open(ontology_filename, 'r') as json_file:
        m_ontology = json.load(json_file)
        check_ontology(m_ontology)
        check_ontology_v2(m_ontology)
