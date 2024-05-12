from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import pandas as pd
import numpy as np
from functools import lru_cache
from rdkit.Chem import Descriptors

@lru_cache(maxsize=None)
def calc_full_pc_descriptors(
    smile: str, 
    descriptors_list: tuple=tuple([i[0] for i in Descriptors._descList if i[0] != 'Ipc']),
    **kwargs
    ) -> dict:

    mol = Chem.MolFromSmiles(smile)
    calculator = MolecularDescriptorCalculator(list(descriptors_list))
    calculated_properties = calculator.CalcDescriptors(mol)

    return {i: j for i, j in zip(list(descriptors_list), calculated_properties)}


@lru_cache(maxsize=None)
def calc_morgan_fp(
    smile: str, 
    r=2, 
    bits=1024, 
    **kwargs) -> dict:

    mol = Chem.MolFromSmiles(smile)
    fp1 = GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=int(r), nBits=int(bits))
    vector = np.array(fp1)
    
    return {str(i): val for i, val in enumerate(vector)}


def calculate_descriptor_frag(mol_smile, fragment_smile, descriptor, **kwargs):
    descriptor_frag = descriptor(fragment_smile, **kwargs)
    return descriptor_frag

def calculate_descriptor_concat(mol_smile, fragment_smile, descriptor, **kwargs):
    descriptor_frag = descriptor(fragment_smile, **kwargs)
    descriptor_frag = {i + '_frag': j for i,j in descriptor_frag.items()}

    descriptor_mol = descriptor(mol_smile, **kwargs)
    descriptor_mol = {i + '_mol': j for i,j in descriptor_mol.items()}
    
    descriptor_frag.update(descriptor_mol)
    return descriptor_frag

def calculate_descriptor_mol(mol_smile, fragment_smile, descriptor, **kwargs):
    descriptor_frag = descriptor(mol_smile, **kwargs)
    return descriptor_frag



def main(args):
    # Sample smile - Just as an example
    smile = "C1CC2=C3C(=CC=C2)C(=CN3C1)[C@H]4[C@@H](C(=O)NC4=O)C5=CNC6=CC=CC=C65"
    atosiban = 'CC[C@H](C)[C@H]1C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CSSCCC(=O)N[C@@H](C(=O)N1)CC2=CC=C(C=C2)OCC)C(=O)N3CCC[C@H]3C(=O)N[C@@H](CCCN)C(=O)NCC(=O)N)CC(=O)N)[C@@H](C)O'
    # All the possible 2D-descriptors that can be computed with rdkit
    descriptors_list = tuple([i[0] for i in Descriptors._descList])

    # Compute descriptors
    descriptors_dict = calc_full_pc_descriptors(atosiban, descriptors_list)
    descriptors_dict = {i + '_frag': j for i,j in descriptors_dict.items()}
    print(pd.DataFrame.from_dict(descriptors_dict, orient="index"))

if __name__ == "__main__":
    main(None)
    