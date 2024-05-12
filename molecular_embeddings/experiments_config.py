from architectures import BinaryClassificationMLP, NeuralFingerprintGNN, MaxPoolingGNN, AddPoolingGNN, TwoLayerGIN
from utils import get_updated_dic
from models import BinaryStandardGNN, BinaryRFC
from config import config

base_GNN_config = {
    'input_size': 13,
    'hidden_channels': 64,
    'batch_norm': False,
    'type_conv': TwoLayerGIN,
    'builder_MLP': BinaryClassificationMLP
}


models = {
    'RFC_FP_eins': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'FP', 'desc_kwargs': {'r': 1}}
        },
    'RFC_FP_zwei': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'FP', 'desc_kwargs': {'r': 2}}
        }, 
    'RFC_FP_drei': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'FP', 'desc_kwargs': {'r': 3}}
        },
    'RFC_FP_vier': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'FP', 'desc_kwargs': {'r': 4}}
        },
    'RFC_PC': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'PC', 'desc_kwargs': {'descriptors_list': config['PC_DESCRIPTORS']}}
        }
    }




'''    
    'GNN_zwei_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 2, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        },
    'GNN_zwei_max': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 2, 'builder_GNN_conv': MaxPoolingGNN}, base_GNN_config)
        }, 
    'RFC_QM': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'QM'}
        },
    'RFC_PC': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'PC', 'desc_kwargs': {'descriptors_list': config['PC_DESCRIPTORS']}}
        }
    'GNN_zwei_no_train_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 2, 'train': False, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        },
    'GNN_zwei_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 2, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        },

    'RFC_FP_eins': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'FP', 'desc_kwargs': {'r': 1}}
        },
    'RFC_FP_zwei': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'FP', 'desc_kwargs': {'r': 2}}
        }, 
    'RFC_FP_drei': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'FP', 'desc_kwargs': {'r': 3}}
        },
    'RFC_FP_vier': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'FP', 'desc_kwargs': {'r': 4}}
        },
    'RFC_PC': {
        'model': BinaryRFC, 
        'model_config': {'DESC_ALGO': 'PC', 'desc_kwargs': {'descriptors_list': config['PC_DESCRIPTORS']}}
        }, 

    'GNN_eins_no_train_max': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 1, 'train': False, 'builder_GNN_conv': MaxPoolingGNN}, base_GNN_config)
        }, 



    'GNN_zwei_no_train_max': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 2, 'train': False, 'builder_GNN_conv': MaxPoolingGNN}, base_GNN_config)
        }, 
    'GNN_drei_no_train_max': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 3, 'train': False, 'builder_GNN_conv': MaxPoolingGNN}, base_GNN_config)
        }, 
    'GNN_vier_no_train_max': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 4, 'train': False, 'builder_GNN_conv': MaxPoolingGNN}, base_GNN_config)
        },
    'GNN_eins_max': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 1, 'builder_GNN_conv': MaxPoolingGNN}, base_GNN_config)
        }, 
    'GNN_zwei_max': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 2, 'builder_GNN_conv': MaxPoolingGNN}, base_GNN_config)
        }, 
    'GNN_drei_max': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 3, 'builder_GNN_conv': MaxPoolingGNN}, base_GNN_config)
        }, 
    'GNN_vier_max': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 4, 'builder_GNN_conv': MaxPoolingGNN}, base_GNN_config)
        }, 

    'GNN_eins_no_train_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 1, 'train': False, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        }, 
    'GNN_zwei_no_train_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 2, 'train': False, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        }, 
    'GNN_drei_no_train_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 3, 'train': False, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        }, 
    'GNN_vier_no_train_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 4, 'train': False, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        },
    'GNN_eins_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 1, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        }, 
    'GNN_zwei_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 2, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        }, 
    'GNN_drei_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 3, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        }, 
    'GNN_vier_add': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 4, 'builder_GNN_conv': AddPoolingGNN}, base_GNN_config)
        }, '''


"""     'GNN_eins_no_train_duvenaud': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 1, 'train': False, 'builder_GNN_conv': NeuralFingerprintGNN, 'decoder_input': 1024}, base_GNN_config)
        }, 
    'GNN_zwei_no_train_duvenaud': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 2, 'train': False, 'builder_GNN_conv': NeuralFingerprintGNN, 'decoder_input': 1024}, base_GNN_config)
        }, 
    'GNN_drei_no_train_duvenaud': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 3, 'train': False, 'builder_GNN_conv': NeuralFingerprintGNN, 'decoder_input': 1024}, base_GNN_config)
        }, 
    'GNN_vier_no_train_duvenaud': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 4, 'train': False, 'builder_GNN_conv': NeuralFingerprintGNN, 'decoder_input': 1024}, base_GNN_config)
        },
    'GNN_eins_duvenaud': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 1, 'builder_GNN_conv': NeuralFingerprintGNN, 'decoder_input': 1024}, base_GNN_config)
        }, 
    'GNN_zwei_duvenaud': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 2, 'builder_GNN_conv': NeuralFingerprintGNN, 'decoder_input': 1024}, base_GNN_config)
        }, 
    'GNN_drei_duvenaud': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 3, 'builder_GNN_conv': NeuralFingerprintGNN, 'decoder_input': 1024}, base_GNN_config)
        }, 
    'GNN_vier_duvenaud': {
        'model': BinaryStandardGNN, 
        'model_config': get_updated_dic({'num_conv_layers': 4, 'builder_GNN_conv': NeuralFingerprintGNN, 'decoder_input': 1024}, base_GNN_config)
        },  """