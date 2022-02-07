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
from sklearn import preprocessing
from datetime import datetime
import scipy
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity


# displays a loading bar in the console
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


# in order to load preprocessed time series data as dataframes, the metadata of each file is contained in the filename
def get_file_info(filename, label=False, histogram=False):
    # remove file extension
    filename = filename.split('.')[0]
    info = filename.split('-')
    if len(info) != 5 and len(info) != 6:
        print("Invalid file name!")
        return None, None, None

    # feature, pca dimensions, rolling mean window
    # example: mot_period-pca-10-rolling-none
    feat, pca_or_bins, roll = info[0], None if not info[2].isdigit() else int(info[2]), None if not info[
        4].isdigit() else int(info[4])
    split_into_weeks = len(info) == 6
    if not label:
        return feat, pca_or_bins, roll, split_into_weeks
    if histogram:
        return f"Histograms of {feat} using {pca_or_bins} bins{f' - rolling mean with window size of {roll}' if roll is not None else ''}" \
               f"{' - split into weeks' if split_into_weeks else ''}"
    else:
        return f"{feat}" \
               f"{f' - reduced to {pca_or_bins} dimensions' if pca_or_bins is not None else ' - not reduced in dimensionality'}" \
               f"{f' - rolling mean with window size of {roll}' if roll is not None else ''}" \
               f"{' - split into weeks' if split_into_weeks else ''}"


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
    features = {'len': [], 'max': [], 'min': [], 'mean': [], 'median': [], 'mode': [], 'std': [], 'skewness': [],
                'kurtosis': [], 'iqr': [], 'count_nan': []}
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
        features['skewness'].append(s.skew(np.array(mot_periods_no_nans)))
        features['kurtosis'].append(s.kurtosis(np.array(mot_periods_no_nans)))
        features['iqr'].append(np.subtract(*np.percentile(mot_periods_no_nans, [75, 25])))
        features['count_nan'].append(((len(mot_periods) - len(mot_periods_no_nans)) / len(mot_periods)) * 100)

    for key in features:
        metadata[key] = features[key]

    metadata.to_pickle('parsed_data/metadata.pkl')


# Load the time series data from each animal, pad it to the same length, interpolate it and optionally calculate
# the rolling mean and/or reduce the dimensionality. pca_fit_after represents the number of examples the PCa is fitted on.
# Without this, too much memory would be needed to transform and save everything at once.
def prepare_timeseries(feature, pca_dims=10, window=None, pca_fit_after=200):
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
    pca_fit_after = min(len(animal_list), pca_fit_after)
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
            if i < pca_fit_after:
                result_list.append(ts_list)
            elif i == pca_fit_after:
                result_list.append(ts_list)
                print("Fitting PCA on the loaded examples...")
                pca.fit(result_list)
                result_list = pca.transform(result_list)
            else:
                a = pca.transform([ts_list])[0]
                result_list = np.vstack([result_list, a])

    data = pd.DataFrame(np.array(result_list).T, columns=columns)
    f = f"preprocessed_data/timeseries/{feature.replace('-', '_')}-pca-{pca_dims or 'none'}-rolling-{window or 'none'}.pkl"
    print(f"Saving result to '{f}'\nThis can now be used when clustering")
    data.to_pickle(f)


# doing the same as above, but split into individual weeks
def prepare_timeseries_per_week(feature, pca_dims=10, window=None, pca_fit_after=200):
    print(f"Preprocessing {feature}, padding the time series to the same size, interpolate missing values"
          f"{f', calculating the rolling mean with a window size of {window}' if window is not None else ''}"
          f"{f', reducing to {pca_dims} using PCA' if pca_dims is not None else ''}...")
    metadata = pd.read_pickle('parsed_data/metadata.pkl')
    result_list = []
    columns = []
    min_max_scaler = preprocessing.MinMaxScaler()
    animal_list = metadata['animal_id'].to_list()
    pca_fit_after = min(len(animal_list), pca_fit_after)
    pca = PCA(n_components=(pca_dims or 2))
    for i, animal_id in enumerate(animal_list):
        progress(i, len(animal_list), f"Loading and padding for each time series - {i}/{len(animal_list)}")
        timeseries = pd.read_pickle(f'parsed_data/timeseries/{animal_id}.pkl')
        timeseries = timeseries[(timeseries['ts'] > datetime.strptime(
            timeseries.iloc[0]['ts'].strftime("%Y-%m-%d") + ' 23:59:59', '%Y-%m-%d %H:%M:%S')) &
                                (timeseries['ts'] < datetime.strptime(
                                    timeseries.iloc[-1]['ts'].strftime("%Y-%m-%d") + ' 00:00:00', '%Y-%m-%d %H:%M:%S'))]
        week_counter = 0
        if len(timeseries.index) == 0:
            continue
        start_of_week = timeseries.iloc[0]['ts']
        timeseries = timeseries[[feature, "ts"]]
        timeseries = timeseries.dropna()
        while True:
            week_counter += 1
            end_of_week = start_of_week + pd.DateOffset(days=7)
            trimmed = timeseries[(timeseries['ts'] > start_of_week) & (timeseries['ts'] < end_of_week)]
            if len(trimmed.index) == 0 or end_of_week > timeseries.iloc[-1]['ts']:
                break

            feature_timeseries = trimmed[feature]

            # pad every time series to the same size and interpolate the missing values
            feature_timeseries = feature_timeseries.reset_index(drop=True)
            feature_timeseries = feature_timeseries.reindex(pd.RangeIndex(start=0, stop=14500, step=1))
            feature_timeseries.interpolate(limit_direction='both', inplace=True)

            # calculated the rolling mean
            if window is not None:
                feature_timeseries = feature_timeseries.rolling(window, min_periods=1).mean()

            ts_list = feature_timeseries.values.reshape(-1, 1)
            ts_list = min_max_scaler.fit_transform(ts_list)[:, 0]
            # ts_list = feature_timeseries.to_list()
            columns.append(animal_id + ' - Week ' + str(week_counter) + " ("+start_of_week.strftime('%d.%m.%Y')+" - "+end_of_week.strftime('%d.%m.%Y')+")")
            start_of_week = end_of_week
            if pca_dims is not None:
                if i < pca_fit_after:
                    result_list.append(ts_list)
                elif i == pca_fit_after and week_counter == 1:
                    # result_list = result_list.tolist()
                    result_list.append(ts_list)
                    print("Fitting PCA on the loaded examples...")
                    pca.fit(result_list)
                    result_list = pca.transform(result_list)
                else:
                    a = pca.transform([ts_list])[0]
                    result_list = np.vstack([result_list, a]).tolist()

    data = pd.DataFrame(np.array(result_list).T, columns=columns)
    f = f"preprocessed_data/timeseries/{feature.replace('-', '_')}-pca-{pca_dims or 'none'}-rolling-{window or 'none'}-split_into_weeks.pkl"
    print(f"Saving result to '{f}'\nThis can now be used when clustering")
    data.to_pickle(f)


# global variables to prevent loading data, clustering data multiple times
kmeans = None
last_input_sig = ""
last_num_clusters = 0
cluster_dataset = []
axes = []
names = []
recent_metadata = np.array([])


def get_signature(feat_num, feat_series, num_clusters):
    return ', '.join((feat_num + ([] if feat_series is None else [feat_series]))) + ' - clusters: ' + str(num_clusters)


def calc_evaluations(metadata, data, assigned_clusters, model, highlight_cat):
    cluster_evals = {}
    if highlight_cat != "none":
        for cate in ['group_id', 'organisation_id', 'organisation_timezone']:
            labels = metadata[cate].to_list()
            # for each datapoint it looks if they are assigned to the same cluster in the label and in the result
            rand = round(metrics.adjusted_rand_score(labels, assigned_clusters), 4)
            # calculated the purity of each cluster by looking how many different labels are assigned to it
            mis = round(metrics.adjusted_mutual_info_score(labels, assigned_clusters), 4)
            cluster_evals[cate] = {'rand': rand, 'mis': mis}

    # for each datapoint it measures the mean distance to all other datapoint in its cluster and
    # the mean distance to all datapoints in the next cluster
    sil = round(metrics.silhouette_score(data, model.labels_, metric='euclidean'), 4)
    cluster_evals['sil'] = sil
    return cluster_evals


def stretch_metadata_into_weeks(original_metadata, names):
    category_list = []
    cats = ['group_id', 'organisation_id', 'organisation_timezone']
    for animal_id_and_week in names:
        a_id = animal_id_and_week.split(' - ')[0]
        new_row = []
        for cate in cats:
            new_row.append(original_metadata[original_metadata['animal_id'] == a_id][cate].values[0])
        category_list.append(new_row)
    return pd.DataFrame(np.array(category_list), columns=cats)


# given the input of the dashboard, calculate the clusters and different evaluations
def calculate_clusters(metadata, feat_num, feat_series, num_clusters, highlight_cat="none"):
    global kmeans, last_input_sig, last_num_clusters, cluster_dataset, axes, names, recent_metadata
    input_sig = get_signature(feat_num, feat_series, num_clusters)

    with open('kmeans_evaluation.json', 'r+') as json_file:
        t = json_file.read()
    kmeans_evaluation = json.loads(t)
    if feat_series is None:
        split_into_weeks = False
    else:
        _, _, _, split_into_weeks = get_file_info(feat_series)
    # of the cluster input is the same as last time the last dataset can be used
    m = metadata
    if last_input_sig != input_sig:
        names = metadata['animal_id'].to_list()
        # if the input is new, create a dataset of the chosen features to cluster later
        axes = []
        cluster_dataset = []
        if split_into_weeks:
            feat_num = []
            input_sig = get_signature(feat_num, feat_series, num_clusters)

        for f in feat_num:
            axes.append(f)
            cluster_dataset.append(metadata[f].to_list())

        if exists(f"preprocessed_data/timeseries/{feat_series}"):
            timeseries = pd.read_pickle(f"preprocessed_data/timeseries/{feat_series}")
            cluster_dataset += timeseries.values.tolist()
            if split_into_weeks:
                names = timeseries.columns.to_list()
                if highlight_cat != "none":
                    if exists("parsed_data/metadata_weekly.pkl"):
                        m = pd.read_pickle("parsed_data/metadata_weekly.pkl")
                        if m.shape[0] != len(names):
                            print(f'\nLoaded metadata ({m.shape[0]}) is not of the same size as data split into weeks '
                                  f'({len(names)})! Recalculating metadata ...\n')
                            if recent_metadata.shape[0] == len(names):
                                print("Using recently-used matadata")
                                m = recent_metadata
                            else:
                                m = stretch_metadata_into_weeks(metadata, names)
                                recent_metadata = m
                    else:
                        m = stretch_metadata_into_weeks(metadata, names)
                        m.to_pickle("parsed_data/metadata_weekly.pkl")

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

    cluster_evals = calc_evaluations(m, cluster_dataset, assigned_clusters, kmeans, highlight_cat)

    kmeans_evaluation[last_input_sig] = cluster_evals

    with open('kmeans_evaluation.json', 'w+') as json_file:
        json.dump(kmeans_evaluation, json_file)

    return cluster_dataset, assigned_clusters, axes, cluster_evals, names, m


def preprocess_histograms(feature, window_enable, window, bins):
    window = None if window_enable is None or window_enable == [] else window
    metadata = pd.read_pickle('parsed_data/metadata.pkl')
    if feature == 'mot_period':
        min_data = metadata.loc[[metadata['min'].idxmax()]]['min'].values[0]
        max_data = metadata.loc[[metadata['max'].idxmax()]]['max'].values[0]
    elif feature == 'mot_pulse_width':
        min_data, max_data = 4, 17
    else:
        min_data, max_data = 0, 1

    result_list = []
    columns = []
    animal_list = metadata['animal_id'].to_list()
    for i, animal_id in enumerate(animal_list):
        progress(i, len(animal_list), f"Calculating histograms for each animal - {i}/{len(animal_list)}")
        timeseries = pd.read_pickle(f'parsed_data/timeseries/{animal_id}.pkl')
        feature_timeseries = timeseries[feature]

        # calculated the rolling mean
        if window_enable:
            feature_timeseries = feature_timeseries.rolling(window, min_periods=1).mean()

        feature_timeseries = [x for x in feature_timeseries.to_list() if str(x) != 'nan']
        n = [0] * bins
        for d in feature_timeseries:
            bin_number = int(bins * ((d - min_data) / (max_data - min_data)))
            n[min(bin_number, (bins - 1))] += 1

        if sum(n) != 0:
            n = [x / sum(n) for x in n]
            result_list.append(n)
            columns.append(animal_id)

    data = pd.DataFrame(np.array(result_list).T, columns=columns)
    f = f"preprocessed_data/histograms/{feature.replace('-', '_')}-bins-{bins or 'none'}-rolling-{window or 'none'}.pkl"
    print(f"Saving result to '{f}'\nThis can now be used when clustering")
    data.to_pickle(f)


def preprocess_histograms_per_week(feature, window_enable, window, bins):
    window = None if window_enable is None or window_enable == [] else window
    metadata = pd.read_pickle('parsed_data/metadata.pkl')
    if feature == 'mot_period':
        min_data = metadata.loc[[metadata['min'].idxmax()]]['min'].values[0]
        max_data = metadata.loc[[metadata['max'].idxmax()]]['max'].values[0]
    elif feature == 'mot_pulse_width':
        min_data, max_data = 4, 17
    else:
        min_data, max_data = 0, 1

    result_list = []
    columns = []
    animal_list = metadata['animal_id'].to_list()
    for i, animal_id in enumerate(animal_list):
        progress(i, len(animal_list), f"Calculating histograms for each animal - {i}/{len(animal_list)}")
        timeseries = pd.read_pickle(f'parsed_data/timeseries/{animal_id}.pkl')
        timeseries = timeseries[(timeseries['ts'] > datetime.strptime(
            timeseries.iloc[0]['ts'].strftime("%Y-%m-%d") + ' 23:59:59', '%Y-%m-%d %H:%M:%S')) &
                                (timeseries['ts'] < datetime.strptime(
                                    timeseries.iloc[-1]['ts'].strftime("%Y-%m-%d") + ' 00:00:00', '%Y-%m-%d %H:%M:%S'))]

        week_counter = 0
        if len(timeseries.index) == 0:
            continue
        start_of_week = timeseries.iloc[0]['ts']
        timeseries = timeseries[[feature, "ts"]]
        timeseries = timeseries.dropna()
        while True:
            week_counter += 1
            end_of_week = start_of_week + pd.DateOffset(days=7)
            trimmed = timeseries[(timeseries['ts'] > start_of_week) & (timeseries['ts'] < end_of_week)]
            if len(trimmed.index) == 0 or end_of_week > timeseries.iloc[-1]['ts']:
                break

            feature_timeseries = trimmed[feature]

            # calculated the rolling mean
            if window_enable:
                feature_timeseries = feature_timeseries.rolling(window, min_periods=1).mean()

            feature_timeseries = [x for x in feature_timeseries.to_list() if str(x) != 'nan']
            n = [0] * bins
            for d in feature_timeseries:
                bin_number = int(bins * ((d - min_data) / (max_data - min_data)))
                n[min(bin_number, (bins - 1))] += 1

            if sum(n) != 0:
                n = [x / sum(n) for x in n]
                result_list.append(n)
                # columns.append(animal_id + ' - Week ' + str(week_counter))
                columns.append(animal_id + ' - Week ' + str(week_counter) + " ("+start_of_week.strftime('%d.%m.%Y')+" - "+end_of_week.strftime('%d.%m.%Y')+")")
            start_of_week = end_of_week

    data = pd.DataFrame(np.array(result_list).T, columns=columns)
    f = f"preprocessed_data/histograms/{feature.replace('-', '_')}-bins-{bins or 'none'}-rolling-{window or 'none'}-split_into_weeks.pkl"
    print(f"Saving result to '{f}'\nThis can now be used when clustering")
    data.to_pickle(f)


def cluster_histograms(metadata, file, method, clusters, highlight_cat):
    global recent_metadata
    cluster_dataset = pd.read_pickle(f"preprocessed_data/histograms/{file}")
    names = cluster_dataset.columns.to_list()
    cluster_dataset = cluster_dataset.to_numpy().T.tolist()

    with open('kmeans_evaluation.json', 'r+') as json_file:
        t = json_file.read()
    kmeans_evaluation = json.loads(t)

    _, _, _, split_into_weeks = get_file_info(file)
    m = metadata
    if split_into_weeks:
        if method == 'kmeans_cos':
            return

    if highlight_cat != "none" and m.shape[0] != len(names):
        if exists("parsed_data/metadata_weekly.pkl"):
            m = pd.read_pickle("parsed_data/metadata_weekly.pkl")
            if m.shape[0] != len(names):
                print(f'\nLoaded metadata ({m.shape[0]}) is not of the same size as data split into weeks '
                      f'({len(names)})! Recalculating metadata ...\n')
                if recent_metadata.shape[0] == len(names):
                    print("Using recently-used matadata")
                    m = recent_metadata
                else:
                    m = stretch_metadata_into_weeks(metadata, names)
                    recent_metadata = m
        else:
            m = stretch_metadata_into_weeks(metadata, names)
            m.to_pickle("parsed_data/metadata_weekly.pkl")

    assigned_labels, model = None, None
    if method == 'kmeans':
        model = KMeans(n_clusters=clusters, random_state=0)
        assigned_labels = model.fit_predict(cluster_dataset)
    elif method == 'kmeans_cos':
        cos_sim = cosine_similarity(cluster_dataset, cluster_dataset, dense_output=False)
        model = KMeans(n_clusters=clusters, random_state=0)
        assigned_labels = model.fit_predict(cos_sim)
        cluster_dataset = cos_sim
    elif method == 'kmeans_avg_cos':
        average_hist = [np.array(cluster_dataset)[:, i].mean() for i in range(len(cluster_dataset[0]))]
        cos_sims = [1 - spatial.distance.cosine(i, average_hist) for i in cluster_dataset]
        model = KMeans(n_clusters=clusters, random_state=0)
        assigned_labels = model.fit_predict(np.array(cos_sims).reshape(-1, 1))
        cluster_dataset = np.array(cos_sims).reshape(-1, 1)
    elif method == 'kmeans_avg_kl':
        average_hist = [np.array(cluster_dataset)[:, i].mean() for i in range(len(cluster_dataset[0]))]
        kl_div = [sum(i) for i in scipy.special.kl_div(cluster_dataset, average_hist)]
        model = KMeans(n_clusters=clusters, random_state=0)
        assigned_labels = model.fit_predict(np.array(kl_div).reshape(-1, 1))
        cluster_dataset = np.array(kl_div).reshape(-1, 1)

    cluster_evals = calc_evaluations(m, cluster_dataset, assigned_labels, model, highlight_cat)
    kmeans_evaluation[get_signature([method], file, clusters)] = cluster_evals

    with open('kmeans_evaluation.json', 'w+') as json_file:
        json.dump(kmeans_evaluation, json_file)

    return cluster_dataset, assigned_labels, names, cluster_evals, m


def find_best_clusters(skip=True, min_clusters=2, max_clusters=6):
    with open('kmeans_evaluation.json', 'r+') as json_file:
        t = json_file.read()
    kmeans_evaluations = json.loads(t)
    metadata = pd.read_pickle('parsed_data/metadata.pkl')

    # change these list to change the tested features
    all_num_feat_to_test = ['max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'iqr', 'count_nan']
    all_ts_feat_to_test = [None,
                           'mot_period-pca-50-rolling-none.pkl',
                           'mot_period-pca-50-rolling-100.pkl',
                           'mot_pulse_width-pca-50-rolling-none.pkl',
                           'mot_pulse_width-pca-50-rolling-100.pkl']
    standalone_test = ['mot_period-pca-50-rolling-none-split_into_weeks.pkl',
                       'mot_period-pca-50-rolling-100-split_into_weeks.pkl',
                       'mot_pulse_width-pca-50-rolling-none-split_into_weeks.pkl',
                       'mot_pulse_width-pca-50-rolling-100-split_into_weeks.pkl']
    histogram_test = ['mot_period-bins-50-rolling-30.pkl',
                      'mot_period-bins-50-rolling-30-split_into_weeks.pkl',
                      'mot_period-bins-50-rolling-none.pkl',
                      'mot_period-bins-50-rolling-none-split_into_weeks.pkl',
                      'mot_pulse_width-bins-50-rolling-30.pkl',
                      'mot_pulse_width-bins-50-rolling-30-split_into_weeks.pkl',
                      'mot_pulse_width-bins-50-rolling-none.pkl',
                      'mot_pulse_width-bins-50-rolling-none-split_into_weeks.pkl',
                      'rum_classification-bins-2-rolling-none.pkl',
                      'rum_classification-bins-2-rolling-none-split_into_weeks.pkl',
                      ]
    histogram_clustering_methods = ['kmeans', 'kmeans_cos', 'kmeans_avg_cos', 'kmeans_avg_kl']

    all_num_feat_combinations = list(
        list(compress(all_num_feat_to_test, mask)) for mask in product(*[[0, 1]] * len(all_num_feat_to_test)))[1:]
    count = 0
    max_iter = len(all_num_feat_combinations) * (max_clusters - min_clusters + 1) * len(all_ts_feat_to_test) + \
               ((max_clusters - min_clusters + 1) * len(standalone_test)) + \
               ((max_clusters - min_clusters + 1) * len(histogram_test) * len(histogram_clustering_methods))
    for num_feat_comb in all_num_feat_combinations:
        for time_series_feat in all_ts_feat_to_test:
            for num_clusters in range(min_clusters, max_clusters + 1):
                count += 1
                signature = get_signature(num_feat_comb, time_series_feat, num_clusters)
                progress(count, max_iter, f'Evaluation clusters of {signature} - {count}/{max_iter}')
                if signature not in kmeans_evaluations and skip:
                    calculate_clusters(metadata, num_feat_comb, time_series_feat, num_clusters, highlight_cat='all')
    for time_series_feat in standalone_test:
        for num_clusters in range(min_clusters, max_clusters + 1):
            count += 1
            signature = get_signature([], time_series_feat, num_clusters)
            progress(count, max_iter, f'Ecvaluation clusters of {signature} - {count}/{max_iter}')
            if signature not in kmeans_evaluations and skip:
                calculate_clusters(metadata, [], time_series_feat, num_clusters, highlight_cat='all')
    for file in histogram_test:
        for method in histogram_clustering_methods:
            for num_clusters in range(min_clusters, max_clusters + 1):
                count += 1
                signature = get_signature([method], file, num_clusters)
                progress(count, max_iter, f'Evaluation clusters of {signature} - {count}/{max_iter}')
                if signature not in kmeans_evaluations and skip:
                    cluster_histograms(metadata, file, method, num_clusters, 'all')


def pretty_print_best_signature(sig, features=None):
    if features is None:
        return sig
    feats, clusters = sig.split(' - clusters: ')
    h = '-bins-' in feats
    feats = feats.split(', ')
    if h:
        return "Histogram Clustering of: '" + get_file_info(feats[1], label=True, histogram=h) + "' using " + \
               features[features['value'] == feats[0]]['label'].values[0] + f' - using {clusters} clusters'
    else:
        for i, _ in enumerate(feats):
            if '.pkl' in feats[i]:
                feats[i] = get_file_info(feats[i], label=True, histogram=False)
            else:
                feats[i] = features[features['value'] == feats[i]]['label'].values[0]
        return "Clustering of: " + ', '.join(feats) + f' - using {clusters} clusters'


def print_best_clusters(top_n_results, rand_weight=0.5, mis_weight=0.5, features=None):
    with open('kmeans_evaluation.json', 'r+') as json_file:
        t = json_file.read()
    kmeans_evaluations = json.loads(t)

    # add categories if more are used
    categories = ['group_id', 'organisation_id', 'organisation_timezone']
    categories_top = {}
    for c in categories:
        categories_top[c] = [{'score': -1, 'sig': ''}] * top_n_results
    sil_top = [{'score': -1, 'sig': ''}] * top_n_results
    for sig in kmeans_evaluations:
        sil_top.append({'score': kmeans_evaluations[sig]['sil'], 'sig': sig})
        sil_top = sorted(sil_top, key=lambda i: i['score'], reverse=True)[:top_n_results]
        for c in categories:
            if c not in kmeans_evaluations[sig]:
                continue
            score = kmeans_evaluations[sig][c]['rand'] * rand_weight + kmeans_evaluations[sig][c]['mis'] * mis_weight
            categories_top[c].append({'score': score, 'sig': sig})
            categories_top[c] = sorted(categories_top[c], key=lambda i: i['score'], reverse=True)[:top_n_results]
    rank = ['', '2nd ', '3rd ']
    output = ""
    for i in range(top_n_results):
        if i > 2:
            r = str(i+1) + 'th '
        else:
            r = rank[i]

        output += (f"------------------------------ The {r}Best Results ------------------------------\n"
                   f"\nThe {r}best result for clusters independent of any labels (according to silhouette score)"
                   f" was archived by using:\n\t\033[4m\033[1m{pretty_print_best_signature(sil_top[i]['sig'], features=features)}"
                   f"\033[0m\n\t\tScore: {sil_top[i]['score']}")
        for c in categories:
            output += (
                f"\n\nThe {r}best result for clusters dependent on the \033[4m\033[1m{c}\033[0m "
                f"(according to adjusted rand score & adjusted mutual information score) were archived by using: "
                f"\n\t\033[4m\033[1m{pretty_print_best_signature(categories_top[c][i]['sig'], features=features)}\033[0m "
                f"\n\t\tAdjusted rand score (weighted {rand_weight}): "
                f"{kmeans_evaluations[categories_top[c][i]['sig']][c]['rand']}"
                f"\n\t\tAdjusted mutual information score (weighted {mis_weight}): "
                f"{kmeans_evaluations[categories_top[c][i]['sig']][c]['mis']}")
        output += '\n\n\n'
    return output


def main():
    step = input("What do you want to do?\n"
                 "Enter '1' to load the initial dataset and reform it into files corresponding to a single animal "
                 "(should be done first) \n"
                 "Enter '2' calculate features like the Standard Deviation, IQR, etc. for the mot_period for each "
                 "animal to use later in clustering (After this step you can use the dashboard apart from "
                 "clustering time series data -> complete step 3)\n"
                 "Enter '3' to pad time series data to the same length, interpolate missing values, calculate "
                 "the rolling mean and save them in separate files for quicker clustering\n"
                 "Enter '4' to create histograms for the time series data. \n"
                 "Enter '5' to loop through different features to find the best clusters\n"
                 "Enter '6' to find the best clusters calculated in step 4\n"
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
        split = input(
            f"\nDo you want to split the data into weeks? (Enter 'y' for yes, leave blank for no)").strip().strip(
            "'").strip('"')
        if split == 'y' or split == 'yes':
            prepare_timeseries_per_week(feature, pca_dims=pca_dims, window=window)
        else:
            prepare_timeseries(feature, pca_dims=pca_dims, window=window)
    elif step == "4":
        possible_features = ['mot_period', 'mot_pulse_width', 'rum_classification']
        feature = input(f"\nWhat feature do you want to preprocess? ({', '.join(possible_features)})"
                        ).strip().strip("'").strip('"')
        feature = feature if feature in possible_features else 'mot_period'
        bins = input(f"\nHow many bins to you want to use? (Leave blank for the default value of 50)").strip()
        bins = 50 if not bins.isdigit() else int(bins)
        window = input(f"\nEnter window size for the rolling mean? (Leave blank to not apply rolling mean)").strip()
        window = None if not window.isdigit() else int(window)
        split = input(
            f"\nDo you want to split the data into weeks? (Enter 'y' for yes, leave blank for no)").strip().strip(
            "'").strip('"')
        if split == 'y' or split == 'yes':
            preprocess_histograms_per_week(feature, True if window is not None else False, window, bins)
        else:
            preprocess_histograms(feature, True if window is not None else False, window, bins)
    elif step == "5":
        find_best_clusters()
    elif step == "6":
        top_n = input(f"\nHow many results do you want to display? (Leave blank for the default value of 5)").strip()
        top_n = 5 if not top_n.isdigit() else int(top_n)
        print(print_best_clusters(top_n))
    else:
        print("Invalid Input!")


if __name__ == '__main__':
    # main()
    find_best_clusters()
