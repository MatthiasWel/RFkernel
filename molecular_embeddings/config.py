import os.path as osp
import torch
from  multiprocessing import cpu_count
from rdkit.Chem import Descriptors
from fragmentize.fragmentize_mol import compute_BRICS_fragments, compute_MAC_fragments 
from descriptor.descriptors import calc_full_pc_descriptors, calc_morgan_fp, calculate_descriptor_frag, calculate_descriptor_concat,calculate_descriptor_mol
config = {}

config['BASE_DIR'] = osp.dirname(osp.realpath(__file__))


config['N_CPUS'] = cpu_count() - 1

config['FRAGMENTATION_ALGORITHMS'] = {
    'MAC': compute_MAC_fragments,
    'BRICS': compute_BRICS_fragments
}

config['PC_DESCRIPTORS'] = tuple([i[0] for i in Descriptors._descList if i[0] not in  ('Ipc', 'BertzCT')]) # very high values
config['DESCRIPTION_ALGORITHMS'] = {
    'PC': calc_full_pc_descriptors,
    'FP': calc_morgan_fp
}

config['DESCRIPTION_STRATEGIES'] = {
    'FRAG': calculate_descriptor_frag,
    'CONCAT': calculate_descriptor_concat,
    'MOL': calculate_descriptor_mol
}

config['ATOM_MASTER_MAPPING'] = {0: 0, 6: 1, 8: 2, 7: 3, 17: 4, 16: 5, 9: 6, 35: 7, 15: 8, 53: 9, 33: 10}
config['TMP'] = '/data/local/tmp'
config['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    import pandas as pd
    pd.options.display.max_colwidth = 100
    print(pd.DataFrame.from_dict(config, orient='index'))