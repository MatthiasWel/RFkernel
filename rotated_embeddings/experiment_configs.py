from MNIST_conv import ResNetMNIST, MNISTDataModule
from MNIST_GNN import LitGNN, MNISTSuperpixelsDataModule
from MNIST_sequence import LSTM_MNIST, SequenceDatamodule 
from architectures_GNN import TwoLayerGIN, AddPoolingGNN, BinaryClassificationMLP

experiment_configs = {
    'Sequence': {
        'model': LSTM_MNIST, 
        'datamodule': SequenceDatamodule, 
        'model_config': {'input_size': 4, 'hidden_channels': 32}
        },
    'ResNet': {
        'model': ResNetMNIST, 
        'datamodule': MNISTDataModule,
        'model_config': {}
        },
    'GNN': {
        'model': LitGNN, 
        'datamodule': MNISTSuperpixelsDataModule,
        'model_config': {
            'input_size': 3,
            'hidden_channels': 32,
            'batch_norm': True,
            'type_conv': TwoLayerGIN,
            'builder_GNN_conv': AddPoolingGNN,
            'builder_MLP': BinaryClassificationMLP,
            'num_conv_layers': 4, 
            'n_classes': 10
            }
        },
}
