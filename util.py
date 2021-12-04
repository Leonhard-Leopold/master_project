import numpy as np
import pandas as pd
import os
import sys
import json
from scipy import stats as s
import math
from sklearn.cluster import KMeans
from sklearn import metrics
from itertools import compress, product
from sklearn.decomposition import PCA
from os.path import exists
from dash import html
from sklearn import preprocessing


# displays a loading bar in the console
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


# in order to load preprocessed time series data as dataframes, the metadata of each file is contained in the filename
def get_file_info(filename, label=False):
    # remove file extension
    filename = filename.split('.')[0]
    info = filename.split('-')
    if len(info) != 5:
        print("Invalid file name!")
        return None, None, None

    # feature, pca dimensions, rolling mean window
    # example: mot_period-pca-10-rolling-none
    feat, pca, roll = info[0], None if not info[2].isdigit() else int(info[2]), None if not info[4].isdigit() else int(
        info[4])
    if label:
        return f"{feat}{f' - rolling mean with window size of {roll}' if roll is not None else ''}" \
               f"{f' - reduced to {pca} dimensions' if pca is not None else ' - not reduced in dimensionality'}"
    else:
        return feat, pca, roll


# the data was provided in parquet files. However, the data was split in a way that there is motility period data from
# multiple animals in a single file. Additionally, some of time series data of a single animal is split among multiple
# of these files. Thus, they needed to be spilt into a single file for each animal.
def parse_data():
    used_ids = []
    if not os.path.exists('parsed_data/timeseries/'):
        os.makedirs('parsed_data/timeseries/')
    # look at every file that is not the metadata file
    files = [f for f in os.listdir("original_data/") if 'metadata' not in f]
    for i, filename in enumerate(files):
        progress(i, len(files), f"Loading and parsing time series files - {i}/{len(files)}")
        df = pd.read_parquet("original_data/" + filename)
        ids = np.unique(np.array(df['animal_id'].to_list()))
        for index, id in enumerate(ids):
            new_vals = df.loc[df['animal_id'] == id].reset_index(drop=True)
            # the new data is saved as a pickle file since it is fast to read and write and is also smaller in size
            f = "parsed_data/timeseries/" + id + ".pkl"
            if id not in used_ids:
                used_ids.append(id)
                new_vals = new_vals.reset_index(drop=True)
                new_vals.to_pickle(f)
            else:
                old_vals = pd.read_pickle(f)
                old_vals.append(new_vals, ignore_index=True)
                old_vals = old_vals.reset_index(drop=True)
                old_vals.to_pickle(f)

    # the metadata is filled since most of the column will not be useful while clustering
    metadata = pd.read_parquet("original_data/metadata.parquet")
    del metadata['name']
    del metadata['official_id']
    del metadata['current_device_id']
    del metadata['insertTime']
    del metadata['group_feed']
    del metadata['archived']
    del metadata['display_name']
    del metadata['mark']
    metadata = metadata[(metadata['animal_id'].isin(used_ids))]
    metadata.to_pickle("parsed_data/metadata.pkl")


# calculate certain feature of the time series data of each animal and save to use for clustering later
def calculate_features():
    metadata = pd.read_pickle('parsed_data/metadata.pkl')
    features = {'len': [], 'max': [], 'min': [], 'mean': [], 'median': [], 'mode': [], 'std': [], 'iqr': [],
                'count_nan': []}
    animal_list = metadata['animal_id'].to_list()
    for i, animal_id in enumerate(animal_list):
        progress(i, len(animal_list), f"Calculation multiple features for each time series - {i}/{len(animal_list)}")
        timeseries = pd.read_pickle(f'parsed_data/timeseries/{animal_id}.pkl')
        mot_periods = timeseries['mot_period'].to_list()
        mot_periods_no_nans = [value for value in mot_periods if not math.isnan(value)]
        features['len'].append(len(mot_periods))
        features['max'].append(max(mot_periods_no_nans))
        features['min'].append(min(mot_periods_no_nans))
        features['mean'].append(np.mean(mot_periods_no_nans))
        features['median'].append(np.median(mot_periods_no_nans))
        features['mode'].append(float(s.mode(mot_periods_no_nans)[0]))
        features['std'].append(np.std(mot_periods_no_nans))
        features['iqr'].append(np.subtract(*np.percentile(mot_periods_no_nans, [75, 25])))
        features['count_nan'].append(((len(mot_periods) - len(mot_periods_no_nans)) / len(mot_periods)) * 100)

    for key in features:
        metadata[key] = features[key]

    metadata.to_pickle('parsed_data/metadata.pkl')


# Load the time series data from each animal, pad it to the same length, interpolate it and optionally calculate
# the rolling mean and/or reduce the dimensionality. num_pca_fit represents the number of examples the PCa is fitted on.
# Without this, too much memory would be needed to transform and save everything at once.
def prepare_timeseries(feature, pca_dims=10, window=None, num_pca_fit=200):
    print(f"Preprocessing {feature}, padding the time series to the same size, interpolate missing values"
          f"{f', calculating the rolling mean with a window size of {window}' if window is not None else ''}"
          f"{f', reducing to {pca_dims} using PCA' if pca_dims is not None else ''}...")
    metadata = pd.read_pickle('parsed_data/metadata.pkl')
    max_series_id = metadata.loc[[metadata['len'].idxmax()]]['animal_id'].values[0]
    max_length_series = pd.read_pickle(f'parsed_data/timeseries/{max_series_id}.pkl')[feature]

    result_list = []
    columns = []
    min_max_scaler = preprocessing.MinMaxScaler()
    animal_list = metadata['animal_id'].to_list()
    num_pca_fit = min(len(animal_list), num_pca_fit)
    pca = PCA(n_components=(pca_dims or 2))
    for i, animal_id in enumerate(animal_list):
        progress(i, len(animal_list), f"Loading and padding for each time series - {i}/{len(animal_list)}")
        timeseries = pd.read_pickle(f'parsed_data/timeseries/{animal_id}.pkl')
        feature_timeseries = timeseries[feature]
        columns.append(animal_id)

        # pad every time series to the same size and interpolate the missing values
        feature_timeseries = feature_timeseries.reindex(max_length_series.index)
        feature_timeseries.interpolate(limit_direction='both', inplace=True)

        # calculated the rolling mean
        if window is not None:
            feature_timeseries = feature_timeseries.rolling(window, min_periods=1).mean()

        ts_list = feature_timeseries.values.reshape(-1, 1)
        ts_list = min_max_scaler.fit_transform(ts_list)[:, 0]

        if pca_dims is not None:
            if i < num_pca_fit:
                result_list.append(ts_list)
            elif i == num_pca_fit:
                result_list.append(ts_list)
                print("Fitting PCA on the loaded examples...")
                pca.fit(result_list)
                result_list = pca.transform(result_list)
            else:
                a = pca.transform([ts_list])[0]
                result_list = np.vstack([result_list, a])

    data = pd.DataFrame(np.array(result_list).T, columns=columns)
    f = f"preprocessed_data/{feature.replace('-', '_')}-pca-{pca_dims or 'none'}-rolling-{window or 'none'}.pkl"
    print(f"Saving result to '{f}'\nThis can now be used when clustering")
    data.to_pickle(f)


# global variables to prevent loading data, clustering data multiple times
kmeans = None
last_input_sig = ""
last_num_clusters = 0
cluster_dataset = []
axes = []


def get_signature(feat_num, feat_series, num_clusters):
    a = ', '.join((feat_num + ([] if feat_series is None else [feat_series]))) + ' - clusters: ' + str(num_clusters)
    return ', '.join((feat_num + ([] if feat_series is None else [feat_series]))) + ' - clusters: ' + str(num_clusters)


# given the input of the dashboard, calculate the clusters and different evaluations
def calculate_clusters(metadata, feat_num, feat_series, num_clusters):
    global kmeans, last_input_sig, last_num_clusters, cluster_dataset, axes
    input_sig = get_signature(feat_num, feat_series, num_clusters)

    with open('kmeans_evaluation.json', 'r+') as json_file:
        t = json_file.read()
    kmeans_evaluation = json.loads(t)

    # of the cluster input is the same as last time the last dataset can be used
    if last_input_sig != input_sig:
        # if the input is new, create a dataset of the chosen features to cluster later
        axes = []
        cluster_dataset = []
        for f in feat_num:
            axes.append(f)
            cluster_dataset.append(metadata[f].to_list())

        if exists(f"preprocessed_data/{feat_series}"):
            timeseries = pd.read_pickle(f"preprocessed_data/{feat_series}")
            cluster_dataset += timeseries.values.tolist()

        cluster_dataset = np.array(cluster_dataset).T

    # if the cluster input string and number of clusters is the same as last time, the model can be reused
    if last_input_sig == input_sig and last_num_clusters == num_clusters:
        print("Able to use model again!")
        assigned_clusters = kmeans.predict(cluster_dataset)
    else:
        # fit K-Means model and predict the cluster assigned to each datapoint
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        assigned_clusters = kmeans.fit_predict(cluster_dataset)

    # safe the current setup
    last_input_sig = input_sig
    last_num_clusters = num_clusters

    cluster_evals = {}
    for cate in ['group_id', 'organisation_id', 'organisation_timezone']:
        labels = metadata[cate].to_list()
        # for each datapoint it looks if they are assigned to the same cluster in the label and in the result
        rand = round(metrics.adjusted_rand_score(labels, assigned_clusters), 4)
        # calculated the purity of each cluster by looking how many different labels are assigned to it
        mis = round(metrics.adjusted_mutual_info_score(labels, assigned_clusters), 4)
        cluster_evals[cate] = {'rand': rand, 'mis': mis}

    # for each datapoint it measures the mean distance to all other datapoint in its cluster and
    # the mean distance to all datapoints in the next cluster
    sil = round(metrics.silhouette_score(cluster_dataset, kmeans.labels_, metric='euclidean'), 4)
    cluster_evals['sil'] = sil
    kmeans_evaluation[last_input_sig] = cluster_evals

    with open('kmeans_evaluation.json', 'w+') as json_file:
        json.dump(kmeans_evaluation, json_file)

    return cluster_dataset, assigned_clusters, axes, cluster_evals


def find_best_clusters(skip=True, min_clusters=3, max_clusters=6):
    with open('kmeans_evaluation.json', 'r+') as json_file:
        t = json_file.read()
    kmeans_evaluations = json.loads(t)
    metadata = pd.read_pickle('parsed_data/metadata.pkl')

    # change these list to change the tested features
    all_num_feat_to_test = ['len', 'mean', 'median', 'std', 'iqr']
    all_ts_feat_to_test = [None, 'mot_period-pca-10-rolling-5.pkl', 'mot_period-pca-10-rolling-none.pkl',
                           'mot_pulse_width-pca-10-rolling-5.pkl', 'mot_pulse_width-pca-10-rolling-none.pkl']

    all_num_feat_combinations = list(
        list(compress(all_num_feat_to_test, mask)) for mask in product(*[[0, 1]] * len(all_num_feat_to_test)))[1:]
    count = 0
    max_iter = len(all_num_feat_combinations) * (max_clusters - min_clusters + 1) * len(all_ts_feat_to_test)
    for num_feat_comb in all_num_feat_combinations:
        for time_series_feat in all_ts_feat_to_test:
            for num_clusters in range(min_clusters, max_clusters + 1):
                count += 1
                signature = get_signature(num_feat_comb, time_series_feat, num_clusters)
                progress(count, max_iter, f'Evaluation clusters of {signature} - {count}/{max_iter}')
                if signature not in kmeans_evaluations and skip:
                    calculate_clusters(metadata, num_feat_comb, time_series_feat, num_clusters)


def print_best_cluster(rand_weight=0.5, mis_weight=0.5):
    with open('kmeans_evaluation.json', 'r+') as json_file:
        t = json_file.read()
    kmeans_evaluations = json.loads(t)

    # add categories if more are used
    categories = ['group_id', 'organisation_id', 'organisation_timezone']
    categories_max = {}
    for c in categories:
        categories_max[c] = {'max': -1, 'sig': ''}
    max_without_cat, max_without_cat_sig = -1, ""
    for sig in kmeans_evaluations:
        if kmeans_evaluations[sig]['sil'] >= max_without_cat:
            max_without_cat = kmeans_evaluations[sig]['sil']
            max_without_cat_sig = sig
        for c in categories:
            score = kmeans_evaluations[sig][c]['rand'] * rand_weight + kmeans_evaluations[sig][c]['mis'] * mis_weight
            if score >= categories_max[c]['max']:
                categories_max[c]['max'] = score
                categories_max[c]['sig'] = sig

    print(f"\n\nThe best clusters independent of any labels (according to silhouette score) were archived by using\n{max_without_cat_sig}\nwith a score of: {max_without_cat}"
          f"\n")
    for c in categories:
        print(f"The best clusters dependent on the {c} (according to adjusted rand score & "
              f"adjusted mutual information score) were archived by using \n{categories_max[c]['sig']} \nwith a score of\n"
              f"adjusted rand score (weighted {rand_weight}): {kmeans_evaluations[categories_max[c]['sig']][c]['rand']}\n"
              f"adjusted mutual information score (weighted {mis_weight}): {kmeans_evaluations[categories_max[c]['sig']][c]['mis']}\n")


def main():
    step = input("What do you want to do?\n"
                 "Enter '1' to load the initial dataset and reform it into files corresponding to a single animal "
                 "(should be done first) \n"
                 "Enter '2' calculate features like the Standard Deviation, IQR, etc. for the mot_period for each "
                 "animal to use later in clustering (After this step you can use the dashboard apart from "
                 "clustering time series data -> complete step 3)\n"
                 "Enter '3' to pad time series data to the same length, interpolate missing values, calculate "
                 "the rolling mean and save them in separate files for quicker clustering\n"
                 "Enter '4' to loop through different features to find the best clusters\n"
                 "Enter '5' to find the best clusters calculated in step 4\n"
                 "(Use the dashboard to display the clusters):").strip().strip("'").strip('"')

    if step == "1":
        parse_data()
    elif step == "2":
        calculate_features()
    elif step == "3":
        possible_features = ['mot_period', 'mot_pulse_width', 'rum_classification']
        feature = input(f"\nWhat feature do you want to preprocess? ({', '.join(possible_features)})"
                        ).strip().strip("'").strip('"')
        feature = feature if feature in possible_features else 'mot_period'
        pca_dims = input(f"\nReduce result to how many dimensions? (Leave blank to not reduce)").strip()
        pca_dims = None if not pca_dims.isdigit() else int(pca_dims)
        window = input(f"\nEnter window size for the rolling mean? (Leave blank to not apply rolling mean)").strip()
        window = None if not window.isdigit() else int(window)
        prepare_timeseries(feature, pca_dims=pca_dims, window=window)
    elif step == "4":
        find_best_clusters()
    elif step == "5":
        print_best_cluster()
    else:
        print("Invalid Input!")


if __name__ == '__main__':
    main()
