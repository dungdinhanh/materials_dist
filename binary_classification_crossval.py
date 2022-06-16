import pandas as pd
from pycaret.classification import *
import numpy as np
from pycaret.utils import check_metric


def replicate(samples, n=5):
    samples = samples.loc[samples.index.repeat(n)]
    return samples
#

def gaussian_noise_adding(samples, alpha=0.001):
    columns = samples.columns
    print(columns)
    values_columns = samples[columns[0:-1]]
    label_columns = samples[columns[-1]]
    # print(values_columns)
    # print(label_columns)




if __name__ == '__main__':
    dataset=pd.read_csv('hdpe_ldpe_original.csv')
    n_rows= len(dataset.index)
    list_indexes = dataset.index.to_numpy()
    np.random.shuffle(list_indexes)
    n_fold = int(n_rows/4)
    list_values = []
    for i in range(n_fold):
        val_indexes = list_indexes[i*4: i *4 + 4]
        data_unseen = dataset.iloc[val_indexes]
        data = dataset.drop(val_indexes)
        data.reset_index(inplace=True, drop=True)
        data_unseen.reset_index(inplace=True, drop=True)
        print(data.shape)
        print(data_unseen.shape)
        data = replicate(data, 5)
        # print(data.values)

        data = gaussian_noise_adding(data, 0.001)
        exit(0)

        exp_clf101 = setup(data=data, target='output', session_id=123, silent=True)
        best_model = compare_models()

        print(best_model)
        # tuned_model = tune_model(best_model)
        # final_model = finalize_model(tuned_model)

        dec_tree = create_model('dt', fold=5, round=2)
        print(dec_tree)
        tuned_model = tune_model(dec_tree)
        final_model = finalize_model(tuned_model)

        unseen_predictions = predict_model(final_model, data=data_unseen)
        # print(unseen_predictions)
        unseen_predictions.head()

        cm = check_metric(unseen_predictions['output'], unseen_predictions['Label'], metric='Accuracy')
        print(cm)
        list_values.append(cm)
    print(list_values)
    exit(0)

    # dataset
    data = dataset.sample(frac=0.7, random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(inplace=True, drop=True)
    data_unseen.reset_index(inplace=True, drop=True)
    print('Data for Modeling: ' + str(data.shape))
    print('Unseen Data For Predictions: ' + str(data_unseen.shape))
    print('original data: ' + str(dataset.shape))
    # augment data
    # replicate
    data = replicate(data, 5)
    # print(data.values)

    exp_clf101 = setup(data = data, target = 'output', session_id=123)
    best_model = compare_models()

    print(best_model)
    # tuned_model = tune_model(best_model)
    # final_model = finalize_model(tuned_model)

    dec_tree = create_model('dt', fold=5, round=2)
    print(dec_tree)
    tuned_model = tune_model(dec_tree)
    final_model = finalize_model(tuned_model)

    unseen_predictions = predict_model(final_model, data=data_unseen)
    # print(unseen_predictions)
    unseen_predictions.head()


    cm = check_metric(unseen_predictions['output'], unseen_predictions['Label'], metric='Accuracy')
    print(cm)
    #%%

    # models()