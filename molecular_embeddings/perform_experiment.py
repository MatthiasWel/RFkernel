import tdc
import tdc.single_pred
import random
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import calendar
import time
import numpy as np
import torch
from scipy.io import savemat
import os
from experiments_config import models

def main():
    current_GMT = time.gmtime()
    start_time_stamp = f'{current_GMT.tm_year}{current_GMT.tm_mon}{current_GMT.tm_mday}_{calendar.timegm(current_GMT)}'
    for base in ('datasets', 'embeddings', 'models', 'tensorboard'):
        path = os.path.join(base, start_time_stamp)
        os.mkdir(path) 

    cyp_adme = [dataset_name for dataset_name in tdc.metadata.adme_dataset_names if  'cyp' in dataset_name]
    n_random_splits = 5
    envs = ('train', 'valid', 'test')
    
    results = pd.DataFrame({'MODEL': [], 'DATASET': [], 'SEED': [], 'MCC_TEST': []})
    try:
        for dataset in cyp_adme:
            print(dataset)
            for i in range(n_random_splits):
                random_seed = random.randint(1, 100)
                print(random_seed)
                dataset_name = f'{random_seed}_{dataset}'
                data = tdc.single_pred.ADME(name=dataset).get_split(seed=42) # random_seed
                for env in envs:
                    df = data[env]
                    assert 'Drug' in df.columns, f'There is a problem in {env}_{dataset} with \"Drug\"'
                    assert 'Y' in df.columns, f'There is a problem in {env}_{dataset} with \"Y\"'
                    df.rename({"Drug":'SMILES', 'Y':'LABEL'}, axis=1, inplace=True)
                    df.to_csv(f'datasets/{start_time_stamp}/{env}_{dataset_name}.csv', index=False)
                    df['SET_TYPE'] = env
                
                train_data = pd.concat([data['train'], data['valid']]).reset_index(drop=True)
                train_data = train_data.drop(columns=['Drug_ID']).drop_duplicates(subset=['SMILES']).reset_index(drop=True)
                test_data = data['test'].drop(columns=['Drug_ID']).drop_duplicates(subset=['SMILES']).reset_index(drop=True)
                y = test_data.LABEL
                for model_type, model_description in models.items():
                    print(model_type)
                    current_GMT = time.gmtime()
                    time_stamp = calendar.timegm(current_GMT)
                    model_config = model_description['model_config']
                    model_config.update({'tensorboard_date': start_time_stamp})
                    
                    model = model_description['model'](
                        f'{dataset_name}', 
                        f'v{time_stamp}', 
                        model_config
                        )
                    if model_config.get('train', True):
                        model.fit(train_data)

                    embedding = model.embedding(test_data)

                    # train_gram = model.gram_rf(train_data, train_data) # train, test too expensive to calc N-large x N_large
                    test_gram_rf = model.gram_rf(train_data, test_data)
                    test_gram_linear = model.linear_gram(test_data)
                    test_gram_rbf_02 = model.rbf_gram(test_data, 0.2)
                    test_gram_rbf_04 = model.rbf_gram(test_data, 0.4)
                    test_gram_rbf_08 = model.rbf_gram(test_data, 0.8)

                    y_hat = model.predict(test_data)
                    mcc_model = matthews_corrcoef(y, np.round(y_hat))
                    mcc_rfc = matthews_corrcoef(y, model.rfc.predict(embedding))
                    mcc_train_rfc = model.rfc_train_fit

                    current_res = pd.DataFrame({
                        'MODEL': [model_type], 
                        'DATASET': [dataset], 
                        'SEED': [random_seed], 
                        'MCC_TEST': [mcc_model],
                        'MCC_TEST_RFC': [mcc_rfc],
                        'MCC_TRAIN_RFC': [mcc_train_rfc],
                        })
                    
                    results = pd.concat([results, current_res])
                    mdic = {
                        'MODEL': model_type, 
                        'DATASET': dataset, 
                        'SEED': random_seed, 
                        'MCC_TEST': mcc_model,
                        'MCC_TEST_RFC': mcc_rfc,
                        'MCC_TRAIN_RFC': mcc_train_rfc,
                        'EMBEDDING': embedding,
                        'PREDICTION': y_hat,
                        'TRUE_LABELS': y,
                        'TEST_GRAM_RF': test_gram_rf,
                        'TEST_GRAM_LINEAR': test_gram_linear,
                        'TEST_GRAM_RBF_02': test_gram_rbf_02,
                        'TEST_GRAM_RBF_04': test_gram_rbf_04,
                        'TEST_GRAM_RBF_08': test_gram_rbf_08
                    }
                    file_name = f'{model_type}_{dataset}_{random_seed}_{time_stamp}'
                    savemat(f'embeddings/{start_time_stamp}/{file_name}.mat', mdic)
                    model.save(f'models/{start_time_stamp}/{file_name}.pt') 

    except Exception as e:
        print(f"Something went wrong: {e}")
    finally:
        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        print(results)
        results.to_csv(f'results/results{start_time_stamp}.csv', index=False)

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('medium')
    main()