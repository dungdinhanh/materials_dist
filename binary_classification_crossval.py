import pandas as pd
from pycaret.classification import *
import numpy as np
from pycaret.utils import check_metric


def replicate(samples, n=5):
    samples = samples.loc[samples.index.repeat(n)]
    return samples
#
def gaussian_noise_adding(samples, nos=4, alpha=0.001):
    columns = samples.columns
    values_columns = samples[columns[0:-1]].to_numpy()
    label_columns = samples[columns[-1]].to_numpy()

    std_values = values_columns.std(0)
    mean_add = np.zeros_like(std_values)
    std_add = std_values * alpha
    cov_add = np.diag(std_add)
    list_new_values = []
    list_new_labels = []
    list_new_values.append(values_columns)
    list_new_labels.append(label_columns)
    for i in range(nos):
        noise_add = np.random.multivariate_normal(mean_add, cov_add, values_columns.shape[0])
        new_values = values_columns + noise_add
        list_new_values.append(new_values)
        list_new_labels.append(label_columns.copy())

    values_gauss = np.concatenate(list_new_values, axis=0)
    labels_gauss = np.concatenate(list_new_labels, axis=0)
    gauss_samples = pd.DataFrame(values_gauss, columns=columns[0:-1])
    gauss_samples['output'] = labels_gauss
    return gauss_samples


def mixup_aug(samples, nos=4, alpha=0.3, type1='hdpe', type2='ldpe'):
    type1_data = samples.loc[samples['output'] == 'hdpe']
    type2_data = samples.loc[samples['output'] == 'ldpe']
    columns = samples.columns

    total_data = samples.shape[0] * nos
    type1_data_value = type1_data[columns[0:-1]].to_numpy()
    type2_data_value = type2_data[columns[0:-1]].to_numpy()
    type1_data_label = type1_data[columns[-1]].to_numpy()
    type2_data_label = type2_data[columns[-1]].to_numpy()
    list_values = []
    list_labels = []

    for i in range(total_data):
        if np.random.uniform() < 0.5:
            # process with type 1 mix up
            type1_pair = type1_data_value[np.random.choice(type1_data_value.shape[0], 2)]
            new_data = type1_pair[0] * alpha + (1 - alpha) * type1_pair[1]
            list_values.append(new_data)
            list_labels.append(type1_data_label[0])
        else:
            type2_pair = type2_data_value[np.random.choice(type2_data_value.shape[0], 2)]
            new_data = type2_pair[0] * alpha + (1 - alpha) * type2_pair[1]
            list_values.append(new_data)
            list_labels.append(type2_data_label[0])

    mixup_values = np.concatenate((samples[columns[0:-1]].to_numpy(), np.vstack(list_values)), axis=0)
    mixup_labels = np.concatenate((samples[columns[-1]].to_numpy(), np.array(list_labels)), axis=0)
    # mixup_labels = np.array(list_labels)
    print(mixup_values.shape)
    print(mixup_labels.shape)
    mixup_samples = pd.DataFrame(mixup_values, columns=columns[0:-1])
    mixup_samples['output'] = mixup_labels
    return mixup_samples
    pass


    # print(std_values)
    # print(std_add)
    # exit(0)
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

        # data = gaussian_noise_adding(data, 4, 0.001)
        # data = mixup_aug(data, 4, 0.3)
        # exit(0)
        # exit(0)

        exp_clf101 = setup(data=data, target='output', session_id=123, silent=True)
        best_model = compare_models()

        # print(best_model)
        tuned_model = tune_model(best_model)
        final_model = finalize_model(tuned_model)
        #
        # dec_tree = create_model('dt', fold=5, round=2)
        # print(dec_tree)
        # tuned_model = tune_model(dec_tree)
        # final_model = finalize_model(tuned_model)

        unseen_predictions = predict_model(final_model, data=data_unseen)
        # print(unseen_predictions)
        unseen_predictions.head()

        cm = check_metric(unseen_predictions['output'], unseen_predictions['Label'], metric='Accuracy')
        list_values.append(cm)
    print(list_values)

