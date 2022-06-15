import pandas as pd
from pycaret.classification import *
import numpy as np
from pycaret.utils import check_metric


def replicate(samples, n=5):
    samples = samples.loc[samples.index.repeat(n)]
    return samples


if __name__ == '__main__':
    dataset=pd.read_csv('hdpe_ldpe_original.csv')
    # dataset
    data = dataset.sample(frac=0.7, random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(inplace=True, drop=True)
    data_unseen.reset_index(inplace=True, drop=True)
    print('Data for Modeling: ' + str(data.shape))
    print('Unseen Data For Predictions: ' + str(data_unseen.shape))
    print('original data: ' + str(dataset.shape))
    # augment data
    # replicate or augmentation here
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