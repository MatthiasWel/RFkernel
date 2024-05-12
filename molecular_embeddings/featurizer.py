import re
import multiprocessing as mp
from collections import defaultdict
from rdkit.Chem import MolFromSmiles
from config import config
from torch.nn.functional import one_hot
import networkx as nx
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
import numpy as np
from torch_geometric.utils.convert import from_networkx
from utils import Difference, flatten, chunks

def GNN_featurize(data):
    data_list = process_raw_data(data).PG_GRAPH.to_list()
    return data_list

def transform_datatypes(geometric_graph):
    geometric_graph.x = geometric_graph.x.type(torch.float32)
    return geometric_graph

def coef_of_variation_node_degree(G):
    node_degrees = [d for n,d in G.degree()]
    return np.std(node_degrees)/np.mean(node_degrees)

def map_median_label(geometric_graph):
    geometric_graph.y = geometric_graph.y.median().type(torch.float32)
    return geometric_graph

def process_raw_data(df):
    return df.assign(
        MOL=lambda x: x.SMILES.map(Chem.MolFromSmiles),
        MW=lambda x: x.MOL.map(ExactMolWt),
        N_RINGS=lambda x: x.MOL.map(lambda mol: len(mol.GetRingInfo().AtomRings())),
        GRAPH=lambda x: x.apply(lambda x: mol_to_nx(mol=x.MOL, label=x.LABEL), axis=1),
        COEF_OF_VAR=lambda x: x.GRAPH.map(coef_of_variation_node_degree),
        PG_GRAPH=lambda x: x.GRAPH.map(from_networkx).map(map_median_label).map(transform_datatypes)
    ) 

def get_all_atom_types(smiles):
    atomtypes = defaultdict(int) # (atomtype: number of occurences)
    atoms = flatten(
        [[Chem.MolFromSmiles(smile).GetAtomWithIdx(i).GetAtomicNum() \
          for i in range(Chem.MolFromSmiles(smile).GetNumAtoms())] \
        for smile in smiles]
    )
    for atom in atoms: 
        atomtypes[atom] += 1
    return atomtypes

def bulk_atom_type_counting(large_smiles, n_chunks=config['N_CPUS']): 
    pool = mp.Pool(config['N_CPUS'])
    dicts = pool.map(
        get_all_atom_types, 
        [list(smiles) for smiles in chunks(large_smiles, n_chunks)]
        )
    result = defaultdict(int)
    for d in dicts:
        for key, value in d.items():
            result[key] += value
    return result


def RF_featurize(dataset, desc_algo, desc_strategy, **kwargs_descriptor):
    dataset['FRAGMENT'] = None
    result = apply_descriptor(
        dataset, 
        desc_algo, 
        desc_strategy, 
        **kwargs_descriptor
        )
    result_filled = result.fillna(0)
    return result_filled

def DeepSet_featurize(dataset, desc_algo='PC', desc_strategy='FRAG', frag_algo='MAC', **kwargs_frag):
    dataset_fragmented = apply_fragmentation(dataset, frag_algo=frag_algo, **kwargs_frag)
    dataset_described = apply_descriptor(
        dataset_fragmented, 
        desc_algo=desc_algo, 
        desc_strategy=desc_strategy, 
        **{'descriptors_list': config['PC_DESCRIPTORS']}
        )
    return descriptor_to_graph_nodes(dataset_described)

def apply_descriptor(dataset, desc_algo, desc_strategy, **kwargs_descriptor):
        fingerprints  = []
        for index, row in tqdm(dataset.iterrows(), leave=False, desc='Descriptor calculation'):
            mol = row['SMILES']
            
            if Chem.MolFromSmiles(mol) is None:
                continue
            fragment = row['FRAGMENT']
            fingerprints.append(
                    list(describe(
                        mol=mol, 
                        fragment=fragment, 
                        desc_algo=desc_algo, 
                        desc_strategy=desc_strategy, 
                        **kwargs_descriptor
                    ).values())
                )
        fingerprints = np.array(fingerprints)
        fingerprints[np.isnan(fingerprints)] = 0
        fingerprints[fingerprints == -np.inf] = - 10 ** 10
        fingerprints[fingerprints == np.inf] = 10 ** 10
        # scaler = StandardScaler() # with_std=False PC19: scaling to (x - mu) / sigma no BertzCT, PC28: scaling to (x - mu)
        # fingerprints = scaler.fit_transform(fingerprints)
        fragments_with_descriptor = pd.concat([dataset, pd.DataFrame(fingerprints)], axis=1)
        # fragments_with_descriptor = fragments_with_descriptor.dropna(axis=0, thresh=10)
        return fragments_with_descriptor #.dropna(axis=1)

def descriptor_to_graph_nodes(data):
    graphs = []
    for smile in tqdm(list(data['SMILES'].unique()), desc='Loading fragments'):
        G = nx.Graph()
        mol = data[data.SMILES == smile]
        G.add_nodes_from([
            (index, {
                 'y': float(row['LABEL']), 
                 'x': row[Difference(row.index, ['LABEL', 'SMILES', 'FRAGMENT', 'SET_TYPE'])].astype(float).fillna(0).tolist()}) \
            for index, row in mol.iterrows() 
        ])
        G.add_edges_from([
            (index, index) \
            for index, _ in mol.iterrows() 
        ])
        
        graphs.append(G)
    data_list = [from_networkx(g) for g in tqdm(graphs, desc='Preprocessing')]
    for graph in data_list:
        graph.y = graph.y.median().type(torch.float32)
        graph.x = graph.x.type(torch.float32)

    return data_list

def apply_fragmentation(dataset, frag_algo='MAC', **kwargs_frag):
    intermediates = []
    for index, row in tqdm(dataset.iterrows(), leave=False, desc='Fragmentation process'):
        smile = row['SMILES']
        label = row['LABEL']
        set_type = row['SET_TYPE']
        fragments = parse_fragments(fragment(smile, algo=frag_algo, **kwargs_frag))
        intermediates.append(
            pd.DataFrame(
            {
                "SMILES": smile, 
                "FRAGMENT": fragments,
                "LABEL": label,
                'SET_TYPE': set_type
            }))
        
    fragmented_mols = pd.concat(intermediates).reset_index(drop=True)
    return fragmented_mols


def fragment(smiles, algo, **kwargs):
    assert algo in config['FRAGMENTATION_ALGORITHMS'].keys(), f"{algo} is no valid fragmentation algorithm"
    return config['FRAGMENTATION_ALGORITHMS'][algo](smiles, False, **kwargs)

def parse_fragments(fragments):
    fragments = [re.sub("\[[0-9]{1,2}\*\]|\(\[[0-9]{1,2}\*\]\)|\[[0-9]{1,2}\*\+\]",'', smile) for smile in fragments]
    
    return [smile for smile in fragments if MolFromSmiles(smile)] # remove smiles that are not molecules

def get_atom_mapping_from_counts(atoms_types_with_counts, threshold=1e-4):
    n_atoms = sum(atoms_types_with_counts.values())
    atom_mapping = {
        atom_number: (i + 1 if count / n_atoms > threshold else 0)
        for i, (atom_number, count) in enumerate(atoms_types_with_counts.items())
    }
    atom_mapping[0] = 0 # if there are no rare atoms
    return atom_mapping

def describe(mol, fragment, desc_algo, desc_strategy, **kwargs):
    assert not mol is None, f"Mol must not be None"
    assert desc_algo in config['DESCRIPTION_ALGORITHMS'].keys(), f"{desc_algo} is no valid description algorithm"
    assert desc_strategy in config['DESCRIPTION_STRATEGIES'].keys(), f"{desc_strategy} is no valid description strategy"
    return config['DESCRIPTION_STRATEGIES'][desc_strategy](
        mol_smile=mol,
        fragment_smile=fragment,
        descriptor=config['DESCRIPTION_ALGORITHMS'][desc_algo],
        **kwargs
    )

def calculate_node_features(atom, atom_mapping=config['ATOM_MASTER_MAPPING']):
    n_atomtypes = len(set(atom_mapping.values()))
    atom_type = one_hot(torch.tensor(atom_mapping.get(atom.GetAtomicNum(), 0)), num_classes=n_atomtypes)
    n_hydrogens = torch.tensor([atom.GetTotalNumHs()])
    n_neighbors = torch.tensor([atom.GetTotalDegree()])
    node_features = torch.cat((atom_type, n_hydrogens, n_neighbors))
    return node_features

def mol_to_nx(mol, label):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   x=calculate_node_features(atom),
                   y=label
                   )
        
    for atom in mol.GetAtoms():
        G.add_edge(atom.GetIdx(),
                   atom.GetIdx(),
                   bond_type=0
                   )
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType()
                   )
    if label:
        G.y = label
    return G