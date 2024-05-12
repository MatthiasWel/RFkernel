import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.tree import _tree
import os
import torch
from torch.nn import BCELoss
import torchmetrics

from architectures import GeneralGNN

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from torch_geometric.loader import DataLoader as PyGDataloader
from torch_geometric.data import Batch as PyGBatch

from itertools import combinations_with_replacement
from multiprocessing import Manager, Process
from config import config
from featurizer import RF_featurize, DeepSet_featurize, GNN_featurize
from utils import Difference, chunks
from joblib import Parallel, delayed, dump, load
import gc
from scipy import sparse
from memory_profiler import profile
from metrics import gram_linear, gram_rbf
from sklearn.metrics import matthews_corrcoef

# fp = open("memoryreport.log", "w+")

PyGBatch.__len__ = lambda self: len(self.y)
class BinaryClassificationModel:
    def __init__(self, name, version, model_config):
        self.model_config = model_config
        self.name = name
        self.version = version

    def featurize(self, data):
        raise NotImplementedError

    def fit(self, train_dataset: pd.DataFrame):
        raise NotImplementedError
    
    def embedding(self, dataset:pd.DataFrame):
        raise NotImplementedError

    def gram_rf(self, *datasets):
        raise NotImplementedError

    def predict(self, test_dataset: pd.DataFrame):
        raise NotImplementedError
    
    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError
    
    def linear_kernel(self, features):
        return gram_linear(features)
    
    def rbf_kernel(self, features, sigma):
        return gram_rbf(features, sigma)

    def next_partition(self, tree, leaves, depth, partition_ids):
        if depth == 0:
            return partition_ids
        new_partition_ids = []
        for node_id in partition_ids:
            if node_id in leaves:
                new_partition_ids += [node_id]                
            else:
                new_partition_ids += [
                    tree.tree_.children_left[node_id],
                    tree.tree_.children_right[node_id]
                    ]
        return self.next_partition(tree, leaves, depth - 1, new_partition_ids)

    def get_partition(self, tree, depth):
        partition_id = [0]
        leaves = [
            node_id for node_id in range(tree.tree_.node_count) if \
            tree.tree_.children_left[node_id] == _tree.TREE_LEAF
            and tree.tree_.children_right[node_id] == _tree.TREE_LEAF
            ]
        depth = depth % np.max(tree.tree_.compute_node_depths())
        return self.next_partition(tree, leaves, depth, partition_id)
      
    def parallel_node_ids(self, forest, tree_indices, n_samples):
        node_ids = Manager().dict()
        def add_node_id(tree_nr, depth, node_ids):
            node_ids[tree_nr] = self.get_partition(forest[tree_nr], depth) 
        jobs = [
            Process(
                target=add_node_id, 
                args=(tree_nr, depth, node_ids)
                ) 
            for tree_nr, depth in zip(tree_indices, np.random.randint(low=0, high=10000, size=n_samples))
            ]
        _ = [p.start() for p in jobs]
        _ = [p.join() for p in jobs]
        id_nodes = {key: val for key, val in node_ids.items()}
        dump(id_nodes, os.path.join(config['TMP'], 'id_nodes'), compress=3)
        del id_nodes
        _ = gc.collect()
   
    def parallel_partition_desc(self, forest, tree_indices, features):
        def add_partition_desc(h):
            path_samples = os.path.join(config['TMP'], str(h))
            samples = load(path_samples)
            forest = load(os.path.join(config['TMP'], 'forest'))
            tree_indices = load(os.path.join(config['TMP'], 'tree_indices'))
            id_nodes = load(os.path.join(config['TMP'], 'id_nodes'))
            out = {
                    tuple(sample): sparse.csr_array(np.concatenate(
                        tuple(forest[tree_nr].decision_path(sample.reshape(1, -1)).toarray()[0, id_nodes.get(tree_nr)]  \
                        for tree_nr in tree_indices),
                        axis=0
                        ).ravel())
                    for sample in samples
                }
            dump(out, path_samples, compress=3)
            del out, samples, forest, id_nodes, tree_indices
            _ = gc.collect()
            return h
        
        hashes_samples = [hash(samples.data.tobytes()) for samples in chunks(features, int(features.shape[0]/config['N_CPUS']))]
        for h, samples in zip(hashes_samples, chunks(features, int(features.shape[0]/config['N_CPUS']))):
            dump(samples, os.path.join(config['TMP'], str(h)), compress=3)
        dump(forest, os.path.join(config['TMP'], 'forest'), compress=3)
        dump(tree_indices, os.path.join(config['TMP'], 'tree_indices'), compress=3)
        
        partition_hashes = Parallel(n_jobs=config['N_CPUS'], verbose=1)(
            delayed(add_partition_desc)(h) for h in hashes_samples)
        
        partition_descriptor = {}
        for h in hashes_samples:
            temp = load(os.path.join(config['TMP'], str(h)))
            partition_descriptor.update(temp)
        dump(partition_descriptor, os.path.join(config['TMP'], 'partition_descriptor'), compress=3)
        del partition_descriptor, temp
        _ = gc.collect()

    def parallel_gram(self, features, n_samples):
        def calc_gram_row(combs):
            partition_descriptor = load(os.path.join(config['TMP'], 'partition_descriptor'))
            gram = np.zeros((len(features), len(features)))
            for comb in combs:
                (index1, sample1), (index2, sample2) = comb
                m1 = partition_descriptor[tuple(sample1)]
                m2 = partition_descriptor[tuple(sample2)]
                kernel = np.sum(m1.indices == m2.indices) / n_samples
                gram[index1, index2] = kernel
                gram[index2, index1] = kernel
            h = hash(gram.data.tobytes())
            dump(gram, os.path.join(config['TMP'], f'gram{h}'), compress=3)
            del partition_descriptor, gram, m1, m2, kernel
            _ = gc.collect()
            return h

        combinations = list(combinations_with_replacement(enumerate(features), 2))
        chunked_combs = chunks(combinations, int(len(combinations)/config['N_CPUS']))
        gram_hashes = Parallel(n_jobs=min(config['N_CPUS'], 25), verbose=1)(
            delayed(calc_gram_row)(combs) for combs in chunked_combs)
        return np.sum([load(os.path.join(config['TMP'], f'gram{h}')) for h in gram_hashes], axis=0)
    
    def rf_kernel(self, rfc, features, n_samples=500):
        tree_indices = np.random.randint(low=0, high=len(rfc.estimators_), size=n_samples)
        forest = rfc.estimators_
        self.parallel_node_ids(forest, tree_indices, n_samples)
        self.parallel_partition_desc(forest=forest, tree_indices=tree_indices, features=features)
        gram = self.parallel_gram(features, n_samples)
        return gram
        

class BinaryRFC(BinaryClassificationModel):
    def __init__(self, name, version, model_config):
        super(BinaryRFC, self).__init__(name, version, model_config)
        self.model = RandomForestClassifier(n_estimators=model_config.get('n_trees', 500), n_jobs=config['N_CPUS'])
        self.desc_kwargs = model_config.get('desc_kwargs', {})

    def featurize(self, data):
        assert set(['SMILES','LABEL', 'SET_TYPE']).issubset(data.columns), f"['SMILES','LABEL', 'SET_TYPE'] must be in {data.columns}"
        processed_data = RF_featurize(
            dataset=data, 
            desc_algo=self.model_config.get('DESC_ALGO', 'PC'), 
            desc_strategy='MOL',
            **self.desc_kwargs
            )
        features = processed_data.loc[:, Difference(processed_data.columns.tolist(), ['LABEL', 'SMILES', 'FRAGMENT', 'SET_TYPE'])].values.astype(np.float32)
        labels = processed_data["LABEL"].values.astype(np.float32)
        return features, labels
    
    def fit(self, train_dataset):
        X, y = self.featurize(train_dataset)
        self.model.fit(X, y)
        self.rfc_train_fit = matthews_corrcoef(y, self.model.predict(X))

    def embedding(self, dataset):
        features, labels = self.featurize(dataset)
        return features

    def gram_rf(self, train, test, *args):
        self.rfc = self.model
        features = self.embedding(test)
        return self.rf_kernel(self.rfc, features)
    
    def linear_gram(self, test):
        features = self.embedding(test)
        return self.linear_kernel(features)
    
    def rbf_gram(self, test, sigma):
        features = self.embedding(test)
        return self.rbf_kernel(features, sigma)

    def predict(self, test_dataset):
        X, y = self.featurize(test_dataset)
        return self.model.predict(X)
    
    def save(self, path):
        pickle.dump(self.model, open(path, "wb"))

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))


class BinaryBaseGNN(BinaryClassificationModel):
    def __init__(self, name, version, model_config):
        super(BinaryBaseGNN, self).__init__(name, version, model_config)
        self.model = LitGNN(self.model_config)
        self.batch_size = self.model_config.get('batch_size', 128)
        self.n_epochs = self.model_config.get('n_epochs', 100)
        self.tensorboard_date = self.model_config.get('tensorboard_date', '')
        self.rfc = None

    def fit(self, train_dataset):
        datamodule = self.featurize(train_dataset)        
        trainer = Trainer(
            max_epochs=self.n_epochs,
            accelerator=config['DEVICE'], 
            callbacks=[
                LearningRateMonitor(logging_interval='step'), 
                ModelCheckpoint(save_top_k=1, monitor="val_MCC_epoch", mode="max", save_last=True)
                ],
            log_every_n_steps=1,
            logger=TensorBoardLogger(
                save_dir=f'tensorboard/{self.tensorboard_date}', 
                name=self.name, 
                version=self.version
                ),
            )

        trainer.fit(
            self.model, 
            datamodule
            )
        
    def _get_feature(self, data):
        raise NotImplementedError
    
    def featurize(self, data):
        assert set(['SMILES','LABEL', 'SET_TYPE']).issubset(data.columns), f"['SMILES','LABEL', 'SET_TYPE'] must be in {data.columns}"
        train = self._get_feature(data[data.SET_TYPE == 'train'])
        val = self._get_feature(data[data.SET_TYPE == 'valid'])
        test = self._get_feature(data[data.SET_TYPE == 'test'])
        datamodule = StandardGNNDataModule(train, val, test, self.batch_size)
        return datamodule
    
    def embedding(self, data):
        data_list = self._get_feature(data)
        dataloader = PyGDataloader(data_list, batch_size=len(data_list), shuffle=False)
        batch = next(iter(dataloader))
        return self.model.embedding(batch).detach().numpy()

    def predict(self, data):
        data_list = self._get_feature(data)    
        dataloader = PyGDataloader(data_list, batch_size=len(data_list), shuffle=False)
        batch = next(iter(dataloader))
        return self.model(batch).detach().numpy()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def gram_rf(self, train, test, n_estimators=500):
        if not self.rfc:
            self.rfc = RandomForestClassifier(n_estimators=n_estimators, n_jobs=config['N_CPUS'])
            features_train = self.embedding(train)
            self.rfc.fit(features_train, train.LABEL)
            self.rfc_train_fit = matthews_corrcoef(train.LABEL, self.rfc.predict(features_train))
        features_test = self.embedding(test)
        return self.rf_kernel(self.rfc, features_test)
    
    def linear_gram(self, test):
        features = self.embedding(test)
        return self.linear_kernel(features)
    
    def rbf_gram(self, test, sigma):
        features = self.embedding(test)
        return self.rbf_kernel(features, sigma)

class BinaryStandardGNN(BinaryBaseGNN):
    def __init__(self, name, version, model_config):
        super(BinaryStandardGNN, self).__init__(name, version, model_config)
    
    def _get_feature(self, data):
        if not data.empty:
            return GNN_featurize(data)
        return


class BinaryDeepSets(BinaryBaseGNN):
    def __init__(self, name, version, model_config): 
        self.model_config = model_config.copy()
        self.model_config['input_size'] = len(config['PC_DESCRIPTORS'])
        super(BinaryDeepSets, self).__init__(name, version, self.model_config)
        self.frag_algo = model_config.get('frag_algo', 'MAC')
        self.kwargs_frag = model_config.get('kwargs_frag', {})

    def _get_feature(self, data):
        if not data.empty:
            features = DeepSet_featurize(
                data, 
                desc_algo=self.model_config.get('DESC_ALGO', 'PC'), 
                desc_strategy='FRAG', 
                frag_algo=self.frag_algo, 
                **self.kwargs_frag
                )
            
            return features
        return
    

class StandardGNNDataModule(LightningDataModule):
    def __init__(
            self,
            train,
            val,
            test, 
            batch_size
            ):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.batch_size = batch_size


    def train_dataloader(self):
        return PyGDataloader(
            self.train, 
            batch_size=self.batch_size, 
            num_workers=config['N_CPUS'],
            shuffle=True
            ) 
    def val_dataloader(self):
        return PyGDataloader(
            self.val, 
            batch_size=len(self.val), 
            num_workers=config['N_CPUS'],
            shuffle=False
            )
    def test_dataloader(self):
        return PyGDataloader(
            self.test, 
            batch_size=len(self.test), 
            num_workers=config['N_CPUS'],
            shuffle=False
            )

class LitGNN(LightningModule):
    def __init__(
            self, 
            model_config,
            lr=1e-4, 
            weight_decay=5e-4, 
            batch_size=128,
            **kwargs
            # momentum=0.9
            ):
        super(LitGNN, self).__init__()
        self.save_hyperparameters()

        # optimizer params
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # metrics
        self.accuracy = torchmetrics.Accuracy('binary')
        self.MCC = torchmetrics.MatthewsCorrCoef('binary')

        # model
        self.loss_function = BCELoss()
        self.GNN = GeneralGNN(model_config)
        
    def embedding(self, data):
        x = self.GNN.embedding(data)
        return x
    
    def forward(self, data):
        x = self.GNN(data)
        return x
    
    def training_step(self, data):
        loss, y_hat, y = self._common_step(data)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        train_accuracy = self.accuracy(y_hat, y)
        self.log("train_accuracy", train_accuracy, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return {'loss':loss, 'y_hat': y_hat, 'y': y}


    def _common_step(self, data):
        y = data.y.flatten()
        y_hat = self.forward(data).flatten()
        loss = self.loss_function(y_hat, y)
        return loss, y_hat, y
    
    
    def validation_step(self, data):
        loss, y_hat, y = self._common_step(data)
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        val_accuracy = self.accuracy(y_hat, y)
        self.log("val_accuracy_epoch", val_accuracy, on_step=False, on_epoch=True, batch_size=self.batch_size)

        val_MCC = self.MCC(y_hat, y)
        self.log("val_MCC_epoch", val_MCC, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return {'loss':loss, 'y_hat': y_hat, 'y': y}

    def predict(self, data):
        loss, y_hat, y = self._common_step(data)
        return {'loss':loss, 'y_hat': y_hat, 'y': y}
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
            ) 
        
        return [
            {"optimizer": optimizer},
           # {"optimizer": optimizer, "lr_scheduler": scheduler}
        ]


