from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import BRICS, MolFromSmiles
import fragmentize.MacFrag as mf

def compute_BRICS_fragments(smile, return_mols=False, minFragmentSize=4, print_invalids=False):
    """
    Fragment a given molecule and return all the associated fragments with BRICS algorithm from rdkit.
    """
    # Returns a dictionary of mol objects of fragments.
    mol = MolFromSmiles(smile)
    
    if not mol:
        print(f"Molecule invalid: {smile}")
        return None
    
    fragments = BRICS.BRICSDecompose(mol, minFragmentSize=minFragmentSize, returnMols=return_mols) 

    
    # If no fragments are obtained print the problematic molecule.
    if not fragments and print_invalids:
        print(f"Failed to fragment: {smile}")
        return [smile]
    
    if not fragments:
        return [smile]
    
    return list(fragments)

def compute_MAC_fragments(smile, return_mols=False, minFragAtoms=8, maxSR=8, maxBlocks=20, print_invalids=False):
    """
    Fragment a given molecule and return all the associated fragments with BRICS algorithm from rdkit.
    """ 
    mol = MolFromSmiles(smile)
    if not mol:
        print(f"Molecule invalid: {smile}")
        return None
    fragments = mf.MacFrag(mol, minFragAtoms=minFragAtoms, maxSR=maxSR, asMols=return_mols, maxBlocks=maxBlocks)  

    # If no fragments are obtained print the problematic molecule.
    if not fragments and print_invalids:
        print(f"Failed to fragment: {smile}")
        return [smile]
    
    if not fragments:
        return [smile]
    
    return list(fragments)


def main(args):
    smile1 = 'CCOc1ccc(C(=O)N2CCCC2C(=O)O)cc1'
    smile2 = 'COc1ccccc1OCCNCC(O)COc1cccc2[nH]c3ccccc3c12'
    smile3 = 'CCCC'
    smile4 = 'CCCC1'
    smiles = [smile1, smile2, smile3, smile4]
    print([compute_BRICS_fragments(smile) for smile in smiles])
    print("*****")
    print([compute_MAC_fragments(smile) for smile in smiles])

if __name__ == "__main__":
    main(None)