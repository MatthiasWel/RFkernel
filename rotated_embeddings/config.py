import os.path as osp
import torch
from  multiprocessing import cpu_count
from rdkit.Chem import Descriptors
config = {}

config['BASE_DIR'] = osp.dirname(osp.realpath(__file__))


config['N_CPUS'] = cpu_count() - 1

config['DATAROOT'] = '/data/local/datasets'

config['ATOM_MASTER_MAPPING'] = {0: 0, 6: 1, 8: 2, 7: 3, 17: 4, 16: 5, 9: 6, 35: 7, 15: 8, 53: 9, 33: 10}
config['TMP'] = '/data/local/tmp'
config['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    import pandas as pd
    pd.options.display.max_colwidth = 100
    print(pd.DataFrame.from_dict(config, orient='index'))