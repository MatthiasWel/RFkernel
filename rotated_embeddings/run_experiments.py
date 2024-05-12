import os
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from scipy.io import savemat
import gc

from random_forest_kernel import rf_kernel
from cka import gram_rbf, gram_linear
from utils import timestamp
from config import config
import warnings
from experiment_configs import experiment_configs

def main():
    start_time_stamp = timestamp()
    for base in ('embeddings', 'models', 'tensorboard'):
        path = os.path.join(base, start_time_stamp)
        os.mkdir(path) 
    results = pd.DataFrame({'MODEL': [], 'MCC_TEST': [], 'SEED': []})
    n_runs = range(5)
    epochs = 200
    try:
        for run in n_runs:
            for exp_name, experiment in experiment_configs.items():
                
                    time_stamp = timestamp()
                    datamodule = experiment['datamodule']()
                    model = experiment['model'](experiment['model_config'])
                    trainer = pl.Trainer(
                        max_epochs=epochs,
                        accelerator=config['DEVICE'],
                        logger=TensorBoardLogger(
                                save_dir=f'tensorboard/{start_time_stamp}/run{run}/',
                                name=exp_name,
                                version=timestamp()
                                )
                        )
                    trainer.fit(
                        model,
                        datamodule
                    )

                    # fit rfc
                    rfc = RandomForestClassifier(n_estimators=500, n_jobs=config['N_CPUS'])
                    train_dataloader = datamodule.train_dataloader()
                    embedding_train = model.embedding(next(iter(train_dataloader))).detach().numpy()
                    labels_train = datamodule.get_labels(train_dataloader).detach().numpy()
                    rfc.fit(embedding_train, labels_train)

                    train_prediction = rfc.predict(embedding_train)
                    mcc_train_rf = matthews_corrcoef(labels_train, train_prediction)

                    # perform tests
                    test_dataloader = datamodule.test_dataloader()
                    embedding_test = model.embedding(next(iter(test_dataloader))).detach().numpy()
                    logits_test_rfc = rfc.predict(embedding_test)
                    labels_test = datamodule.get_labels(test_dataloader).detach().numpy()
                    mcc_rf = matthews_corrcoef(labels_test, logits_test_rfc)

                    loss, y, logits_test_model = model._common_step(next(iter(test_dataloader)))
                    pred_model = (logits_test_model >= 0.5).long().detach().numpy()
                    mcc_model = matthews_corrcoef(labels_test, pred_model)


                    current_res = pd.DataFrame({
                        'MODEL': [exp_name], 
                        'SEED': [run], 
                        'MCC_TEST_RF': [mcc_rf],
                        'MCC_TRAIN_RF': [mcc_train_rf],
                        'MCC_MODEL': [mcc_model]
                        })
                    

                    results = pd.concat([results, current_res])


                    test_gram_rf = rf_kernel(rfc, embedding_test)
                    test_gram_linear = gram_linear(embedding_test)
                    test_gram_rbf_02 = gram_rbf(embedding_test, 0.2)
                    test_gram_rbf_04 = gram_rbf(embedding_test, 0.4)
                    test_gram_rbf_08 = gram_rbf(embedding_test, 0.8)

                    mdic = {
                        'MODEL': exp_name, 
                        'SEED': run,
                        'MCC_TEST_RF': mcc_rf,
                        'PREDICTION_RF': logits_test_rfc,
                        'MCC_MODEL': mcc_model,
                        'PREDICTION_MODEL': pred_model, 
                        'TRUE_LABELS': labels_test,
                        'TEST_GRAM_RF': test_gram_rf,
                        'TEST_GRAM_LINEAR': test_gram_linear,
                        'TEST_GRAM_RBF_02': test_gram_rbf_02,
                        'TEST_GRAM_RBF_04': test_gram_rbf_04,
                        'TEST_GRAM_RBF_08': test_gram_rbf_08
                    }
                    file_name = f'{exp_name}_{run}_{time_stamp}'
                    savemat(f'embeddings/{start_time_stamp}/{file_name}.mat', mdic)
                    torch.save(model.state_dict(), f'models/{start_time_stamp}/{file_name}.pt')
                    del datamodule, model, mdic, file_name, pred_model, \
                        loss, y, logits_test_model, mcc_model, mcc_rf, logits_test_rfc, \
                        rfc, train_dataloader, embedding_train, labels_test, test_gram_linear, \
                        test_gram_rbf_02, test_gram_rbf_04, test_gram_rbf_08, test_gram_rf
                    gc.collect()

    except Exception as e:
        print(f"Something went wrong: {e}")
    finally:
        print(results)
        results.to_csv(f'results/results{timestamp()}.csv', index=False)

            

        

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.multiprocessing.set_sharing_strategy('file_system')
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    main()