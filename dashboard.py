import numpy as np
import pandas as pd
import dash
from dash import html
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc
from dash.dependencies import Input, Output
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import flask
from os.path import exists
from util import calculate_clusters, get_file_info, parse_data, calculate_features, prepare_timeseries, \
    preprocess_histograms, calc_evaluations, prepare_timeseries_per_week, preprocess_histograms_per_week, \
    print_best_clusters, get_signature, cluster_histograms

logs = [html.Div('Logs:', style={'font-family': 'Consolas', 'font-size': '20px', 'font-weight': 'bold'})]
previous_log_length = 0
layouts = {}

# a dataframe of all features to filter them and label the axes of graphs
features = pd.DataFrame([
    ['len', 'Length', 'numerical'],
    ['max', 'Maximum', 'numerical'],
    ['min', 'Minimum', 'numerical'],
    ['mean', 'Mean', 'numerical'],
    ['median', 'Median', 'numerical'],
    ['mode', 'Mode', 'numerical'],
    ['std', 'Standard deviation', 'numerical'],
    ['skewness', 'Skewness', 'numerical'],
    ['kurtosis', 'Kurtosis', 'numerical'],
    ['iqr', 'Interquartile range', 'numerical'],
    ['count_nan', 'Percentage of invalid measurements', 'numerical'],
    ['group_id', 'Group ID of an animal', 'categorical'],
    ['organisation_id', 'Organisation ID of an animal', 'categorical'],
    ['organisation_timezone', 'Timezone the animal is located in', 'categorical'],
    ['tsne1', "t-SNE 1", 'tsne'],
    ['tsne2', "t-SNE 2", 'tsne'],
    ['tsne3', "t-SNE 3", 'tsne'],
    ['kmeans', 'Clustering just using the bins of the histograms', 'method'],
    ['kmeans_cos', 'Clustering the cosine similarity between all of the histograms', 'method'],
    ['kmeans_avg_cos', 'Clustering the cosine similarity between every histogram and the average histogram', 'method'],
    ['kmeans_avg_kl', 'Clustering the KL divergence between every histogram and the average histogram', 'method'],
    ['pca1', "Principal Component 1", 'pca'],
    ['pca2', "Principal Component 2", 'pca'],
    ['pca3', "Principal Component 3", 'pca'],
    ['avg1', "Similarity to the average value", 'avg'],
    ['mot_period', "Motility Period", 'time'],
    ['mot_pulse_width', "Motility Pulse Width", 'time'],
    ['rum_classification', "Rumination Classification", 'time']],
    columns=['value', 'label', 'category'])


def get_label(v):
    return features.loc[features['value'] == v]['label'].values[0]


def start_dashboard():
    # load the calculated features (util.py -> calculate_features())
    if exists('parsed_data/metadata.pkl'):
        metadata = pd.read_pickle('parsed_data/metadata.pkl')
        init_animal_id = metadata['animal_id'].to_list()[0]
    else:
        metadata = pd.DataFrame(['No animal available'], columns=['animal_id'])
        init_animal_id = ""
    # load data for first animal to speed up displaying the graph
    if os.path.exists(f"parsed_data/timeseries/{init_animal_id}.pkl"):
        init_df = pd.read_pickle(f"parsed_data/timeseries/{init_animal_id}.pkl")

        # dict of already loaded Motility Periods in case they need to be displayed again
        already_loaded = {init_animal_id: init_df.to_dict()}
    else:
        init_df = None
        already_loaded = {}

    # start dashboard
    app = dash.Dash()
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True

    # Load all animals ids to display them in the dropdown menus in the time series tabs
    animal_id_options = []
    for animal_id in metadata['animal_id'].to_list():
        animal_id_options.append({'label': animal_id, 'value': animal_id})

    # Get all preprocessed time series data and prepare it for selection in the clustering tab
    preprocessed_timeseries_data_options = [{'label': "No time series", 'value': 'none'}]
    preprocessed_files_timeseries = os.listdir("preprocessed_data/timeseries/")
    for f in preprocessed_files_timeseries:
        lab = get_file_info(f, label=True, histogram=False)
        preprocessed_timeseries_data_options.append({'label': lab, 'value': f})

    preprocessed_histograms_data_options = []
    preprocessed_files_histograms = os.listdir("preprocessed_data/histograms/")
    for f in preprocessed_files_histograms:
        lab = get_file_info(f, label=True, histogram=True)
        preprocessed_histograms_data_options.append({'label': lab, 'value': f})

    original_files_given = len([f for f in os.listdir("original_data/") if 'metadata' not in f]) > 0 \
                           and exists('original_data/metadata.parquet')
    parsed_files = [] if not exists("parsed_data/timeseries/") else os.listdir("parsed_data/timeseries/")
    step1_completed = 'completed' if len(parsed_files) > 0 and len(metadata['animal_id'].to_list()) > 0 else ''

    # Create setup tab layout
    layouts['layout_setup'] = html.Div(id='parent_setup', children=[
        html.H1(children='Setup'),
        html.Div(id='setup_wrapper', children=[
            html.Div(id='step1_wrapper', className=step1_completed, children=[
                html.H3("Step 1: Parsing original data"), html.Span(" - already completed!"),
                html.P("The original files will be separated, filtered and stored in a single file for each animal",
                       className='setup_explanation'),
                html.Div(className='setup_requirements_wrapper', children=[
                    html.Span("Requirements: ", className='req_span'),
                    html.Span("The original data must be placed in the 'original_data' folder.\n"),
                    html.Span(className=f"requirements {'met' if original_files_given else ''}"),
                    html.Span("Enables: ", className='enables_span'),
                    html.Span(" Visualizing time series data, step 2 & step 3"),
                    html.Button("Go!", id="button_step1")
                ])]),
            html.Div(id='step2_wrapper',
                     className='completed' if 'len' in metadata.columns else '', children=[
                    html.H3("Step 2: Calculation numerical features"),
                    html.Span(" - already completed!"),
                    html.P("Numerical Features like the length, standard deviation or IQR are calculated for the "
                           "time series data for each animal.", className='setup_explanation'),
                    html.Div(className='setup_requirements_wrapper', children=[
                        html.Span("Requirements: ", className='req_span'), html.Span("Step 1\n"),
                        html.Span(className=f"requirements {'met' if step1_completed else ''}", id="step2_requirement"),
                        html.Span("Enables: ", className='enables_span'),
                        html.Span("Clustering & creating histograms of these features"),
                        html.Button("Go!", id="button_step2"),
                    ])]),
            html.Div(id='step3_wrapper', className='completed' if len(preprocessed_files_timeseries) else '', children=[
                html.H3("Step 3: Preprocessing time series data"),
                html.Span(" - already completed!"),
                html.P(
                    "The time series data is not of the same length, not normalized and usually too big to use for "
                    "clustering. Here, the data will be preprocessed and stored.", className='setup_explanation'),
                html.P(f"(Currently there are {len(preprocessed_files_timeseries)} already preprocessed files.)",
                       id="step3_num_files"),
                html.Div(className='setup_requirements_wrapper', children=[
                    html.Span("Requirements: ", className='req_span'), html.Span("Step 1\n"),
                    html.Span(className=f"requirements {'met' if step1_completed else ''}", id="step3_requirement"),
                    html.Div([html.P('Feature:'),
                              dcc.Dropdown(id='step3_feature', options=features[features['category'] == 'time'][
                                  ['label', 'value']].to_dict('records')),
                              html.Span('Reduce to # dimensions:'),
                              dcc.Checklist(id='step_3_enable_pca', value=['enabled'],
                                            options=[{'label': 'enable', 'value': 'enabled'}]),
                              dcc.RangeSlider(id="step3_PCA", min=2, max=100, value=[50], marks={2: '2', 100: '100'},
                                              tooltip={"placement": "bottom", "always_visible": True}),
                              html.Span('Rolling mean window:'),
                              dcc.Checklist(id='step_3_enable_rolling',
                                            options=[{'label': 'enable', 'value': 'enabled'}]),
                              dcc.RangeSlider(id="step3_rolling", min=2, max=1000, value=[100],
                                              marks={2: '2', 1000: '1000'},
                                              tooltip={"placement": "bottom", "always_visible": True}),
                              html.Span('Split data into weeks?', style={'font-weight': 'bold'}),
                              dcc.Checklist(id='step_3_enable_weeks',
                                            options=[{'label': 'Split into weeks!', 'value': 'yes'}]),
                              ]),
                    html.Span("Enables: ", className='enables_span'),
                    html.Span("Using time series data for clustering"),
                    html.Button("Go!", id="button_step3"),
                ])]),
            html.Div(children=[
                html.H3("Find best results"),
                html.P("Everytime something was clustered, the resulting scores were saved and written to a file."
                       "Click the button below to display the best results!", className='setup_explanation'),
                html.P('How many results to display?', style={'font-weight': 'bold'}),
                dcc.RangeSlider(id="top_n_results", min=1, max=20, value=[5],
                                marks={1: '1', 20: '20'},
                                tooltip={"placement": "bottom", "always_visible": True}),
                dcc.Textarea(id='results_textarea', disabled=True),
                html.Button("Go!", id="button_results")])
        ])
    ])

    # Create histogram tab layout
    layouts['layout_histogram'] = html.Div(id='parent_histogram', children=[
        html.H1(children='Histogram/Heatmap of a chosen Feature'),
        html.P("Select one feature to generate its histogram.\nSelect two features to generate a heatmap."),
        html.Div(className="half", children=[
            html.H2("Feature 1:"),
            dcc.Dropdown(id='dropdown_histogram1', value='len',
                         options=features[features['category'] == 'numerical'][['label', 'value']].to_dict('records')),
            html.P("Number of Bins:"),
            dcc.RangeSlider(id="slider_histogram1", min=2, max=100, value=[50], marks={2: '2', 100: '100'},
                            tooltip={"placement": "bottom", "always_visible": True})
        ]),
        html.Div(className="half", children=[
            html.H2("Feature 2:"),
            dcc.Dropdown(id='dropdown_histogram2', value='none',
                         options=[{'label': 'No Feature', 'value': 'none'}] +
                                 features[features['category'] == 'numerical'][['label', 'value']].to_dict('records')),
            html.P("Number of Bins:"),
            dcc.RangeSlider(id="slider_histogram2", min=2, max=100, value=[50], marks={2: '2', 100: '100'},
                            tooltip={"placement": "bottom", "always_visible": True})
        ]),
        dcc.Loading(id="loading-icon-1", type="default", children=[html.Div(id="histogram_wrapper",
                                                                            children=[dcc.Graph(id='histogram')])])
    ])

    # Create barchart tab layout
    layouts['layout_barchart'] = html.Div(id='parent_barchart', children=[
        html.H1(children='Barchart of a chosen Feature'),
        dcc.Dropdown(id='dropdown_barchart', value='group_id',
                     options=features[features['category'] == 'categorical'][['label', 'value']].to_dict('records')),
        dcc.Loading(id="loading-icon-2", type="default",
                    children=[html.Div(dcc.Graph(id='barchart'))])
    ])

    # Create time series with rolling mean tab layout
    layouts['layout_timeseries_rolling'] = html.Div(id='parent_timeseries_rolling', children=[
        html.H1(children='Visualizing Time Series Data - Rolling Mean'),
        html.H3("Animal ID:"),
        dcc.Dropdown(id='dropdown_timeseries_rolling_id', options=animal_id_options,
                     value=animal_id_options[0]['value']),
        html.H3("Feature to visualize:"),
        dcc.Dropdown(id='dropdown_timeseries_rolling_feature', value='mot_period',
                     options=features[features['category'] == 'time'][['label', 'value']].to_dict('records')),
        html.Span("Size of Rolling Mean Window:"),
        dcc.Checklist(id='timeseries_rolling_enabled',
                      options=[{'label': 'enable', 'value': 'enabled'}]),
        dcc.RangeSlider(id="slider_timeseries_rolling_window", min=2, max=1000, value=[30],
                        marks={2: '2', 1000: '1000'}, tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Loading(id="loading-icon-4", type="default", children=[html.Div(dcc.Graph(id='timeseries_rolling'))])
    ])

    # Create clustering tab layout
    layouts['layout_clustering'] = html.Div(id='parent_clustering', children=[
        html.H1(children='Clustering'),
        html.Div(className="wrapper", children=[
            html.Div(className="half",
                     children=[html.H3("Numerical Features:", style={'font-weight': 'bold', 'margin-top': '30px'}),
                               dcc.Checklist(id='clustering_features_numerical', value=['mean', 'std'],
                                             options=features[features['category'] == 'numerical'][
                                                 ['label', 'value']].to_dict('records'))]),
            html.Div(className="half",
                     children=[html.H3("Time Series Data:"),
                               dcc.Dropdown(id='clustering_features_timeseries', value='none',
                                            options=preprocessed_timeseries_data_options)]),

            html.Div(className="half", children=[
                html.H3("Number of Clusters:"),
                dcc.RangeSlider(id="clustering_clusters", min=2, max=15, value=[3], marks={2: '2', 15: '15'},
                                tooltip={"placement": "bottom", "always_visible": True})]),
            html.Div(className="half",
                     children=[html.H3("Category to Highlight:"),
                               dcc.Dropdown(id='clustering_highlight_categorical', value='none',
                                            options=[{'label': 'No Feature', 'value': 'none'}] +
                                                    features[features['category'] == 'categorical'][
                                                        ['label', 'value']].to_dict('records'))]),
            html.Div(className="half",
                     children=[html.H3("Display in:"),
                               dcc.RadioItems(id='clustering_dimensions', value=2,
                                              options=[{'label': '1 Dimension', 'value': 1},
                                                       {'label': '2 Dimensions', 'value': 2},
                                                       {'label': '3 Dimensions', 'value': 3}])]),
            html.Div(className="half", children=[
                html.H3("Reduce dimensions (if necessary) using:"),
                dcc.Dropdown(id='clustering_dimensions_method', value='pca',
                             options=[{'label': 'PCA', 'value': 'pca'}, {'label': 't-SNE', 'value': 'tsne'}])]),
            html.Button('Calculate Clusters!', id='cluster_button'),
        ]),
        dcc.Loading(id="loading-icon-5", type="default",
                    children=[html.Div(id='clustering_wrapper',
                                       children=[dcc.Graph(id='clustering')])])
    ])

    # Create barchart tab layout
    layouts['layout_clustering_histogram'] = html.Div(id='parent_clustering_histogram', children=[
        html.H1(children='Clustering of histograms'),
        html.Div(children=[html.H2(children='Histogram of a single animal'),
                           html.P("Animal ID:"),
                           dcc.Dropdown(id='dropdown_clustering_histogram_id', options=animal_id_options,
                                        value=animal_id_options[0]['value']),
                           html.P("Feature to visualize:"),
                           dcc.Dropdown(id='dropdown_clustering_histogram_feature', value='mot_period',
                                        options=features[features['category'] == 'time'][['label', 'value']].to_dict(
                                            'records')),
                           html.Span("Size of Rolling Mean Window:"),
                           dcc.Checklist(id='clustering_histogram_enable_rolling',
                                         options=[{'label': 'enable', 'value': 'enabled'}]),
                           dcc.RangeSlider(id="slider_clustering_histogram_rolling_window", min=2, max=1000, value=[30],
                                           marks={2: '2', 1000: '1000'},
                                           tooltip={"placement": "bottom", "always_visible": True}),
                           html.Span("Number of Bins:"),
                           dcc.RangeSlider(id="slider_clustering_histogram_bins", min=2, max=100, value=[50],
                                           marks={2: '2', 100: '100'},
                                           tooltip={"placement": "bottom", "always_visible": True}),
                           dcc.Loading(id="loading-icon-6", type="default",
                                       children=[html.Div(dcc.Graph(id='clustering_single_histogram'))]), ]),
        html.Div(children=[html.H2(children='Calculating and preprocessing histograms for each animal'),
                           html.P(
                               f"(There are {len(preprocessed_histograms_data_options)} different versions of preprocessed histograms)",
                               id='preprocessed_histograms', className='subtext'),
                           html.P("Feature to use:"),
                           dcc.Dropdown(id='dropdown_clustering_histogram_preprocess_feature', value='mot_period',
                                        options=features[features['category'] == 'time'][['label', 'value']].to_dict(
                                            'records')),
                           html.Span("Size of Rolling Mean Window:"),
                           dcc.Checklist(id='clustering_histogram_preprocess_enable_rolling',
                                         options=[{'label': 'enable', 'value': 'enabled'}]),
                           dcc.RangeSlider(id="slider_clustering_histogram_preprocess_rolling_window", min=2,
                                           max=1000, value=[30], marks={2: '2', 1000: '1000'},
                                           tooltip={"placement": "bottom", "always_visible": True}),
                           html.Span("Number of Bins:"),
                           dcc.RangeSlider(id="slider_clustering_histogram_preprocess_bins", min=2, max=100, value=[50],
                                           marks={2: '2', 100: '100'},
                                           tooltip={"placement": "bottom", "always_visible": True}),
                           html.Span('Split data into weeks?', style={'font-weight': 'bold'}),
                           dcc.Checklist(id='clustering_histogram_preprocess_enable_weeks',
                                         options=[{'label': 'Split into weeks!', 'value': 'yes'}]),
                           html.Button("Preprocess!", id='preprocess_histogram_button'), ]),
        html.Div(children=[html.H2(children='Clustering preprocessed histograms'),
                           html.Div(className="half", children=[
                               html.H3("Histograms of:"),
                               dcc.Dropdown(id='clustering_histogram_file',
                                            options=preprocessed_histograms_data_options, value='none')]),
                           html.Div(className="half", children=[
                               html.H3('Clustering method:'),
                               dcc.Dropdown(id='clustering_histogram_method', value='pca',
                                            options=features[features['category'] == 'method'][
                                                ['label', 'value']].to_dict('records'))]),
                           html.Div(className="half", children=[
                               html.H3("Number of Clusters:"),
                               dcc.RangeSlider(id="slider_clusters_histogram", min=2, max=20, value=[3],
                                               marks={2: '2', 20: '20'},
                                               tooltip={"placement": "bottom", "always_visible": True})]),
                           html.Div(className="half",
                                    children=[html.H3("Category to Highlight:"),
                                              dcc.Dropdown(id='clustering_histogram_highlight_categorical',
                                                           value='none',
                                                           options=[{'label': 'No Feature', 'value': 'none'}] +
                                                                   features[features['category'] == 'categorical'][
                                                                       ['label', 'value']].to_dict('records'))]),
                           html.Div(className="half", children=[
                               html.H3("Display in:"),
                               dcc.RadioItems(id='clustering_histogram_dimensions', value=2,
                                              options=[{'label': '1 Dimension', 'value': 1},
                                                       {'label': '2 Dimensions', 'value': 2},
                                                       {'label': '3 Dimensions', 'value': 3}])]),
                           html.Div(className="half", children=[
                               html.H3("Reduce dimensions using:"),
                               dcc.Dropdown(id='clustering_histogram_dimensions_method', value='pca',
                                            options=[{'label': 'PCA', 'value': 'pca'},
                                                     {'label': 't-SNE', 'value': 'tsne'}])]),
                           html.Button("Cluster!", id='clustering_histogram_button'),
                           dcc.Loading(id="loading-icon-7", type="default",
                                       children=[html.Div(id='clustering_histogram_wrapper',
                                                          children=[dcc.Graph(id='clustering_histogram')])])]),
    ])

    # Create layout of the entire webapp
    app.layout = html.Div([
        html.Link(href='/static/dashboard.css', rel='stylesheet'),
        dcc.Tabs(id="tabs", children=[
            dcc.Tab(label='Setup', value='layout_setup'),
            dcc.Tab(label='Feature Histogram', value='layout_histogram'),
            dcc.Tab(label='Feature Barchart', value='layout_barchart'),
            dcc.Tab(label='Timeseries Visualisation', value='layout_timeseries_rolling'),
            dcc.Tab(label='Clustering', value='layout_clustering'),
            dcc.Tab(label='Clustering Histograms', value='layout_clustering_histogram')
        ], value='layout_setup'),
        html.Div(id='tabs-output'),
        html.Hr(),
        dcc.Interval(id='log-update', interval=1000),
        html.Div(id='log-area', children=[html.Div('Logs:')])
    ])

    # render layout of a tab if it is clicked on
    @app.callback(Output('tabs-output', 'children'), Input('tabs', 'value'))
    def render_content(tab):
        return [layouts[tab]]

    # include stylesheet
    @app.server.route('/static/<path:path>')
    def static_file(path):
        static_folder = os.path.join(os.getcwd(), 'static')
        return flask.send_from_directory(static_folder, path)

    # display log messages if there are new ones
    @app.callback(Output('log-area', 'children'), [Input('log-update', 'n_intervals')])
    def update_logs(interval):
        global previous_log_length, logs
        if len(logs) == previous_log_length:
            return dash.no_update
        else:
            previous_log_length = len(logs)
            return logs

    # parse the original data into individual files
    @app.callback([Output(component_id='step1_wrapper', component_property='className'),
                   Output(component_id='step2_requirement', component_property='className'),
                   Output(component_id='step3_requirement', component_property='className')],
                  [Input(component_id='button_step1', component_property='n_clicks')])
    def setup_step1(_):
        if dash.callback_context.triggered[0]['prop_id'].split('.')[0] != 'button_step1':
            return dash.no_update
        if not original_files_given:
            logs.append(html.Div('Please place the original files in the "original_files" folder!',
                                 style={'color': 'red', 'font-weight': 'bold'}))
            return dash.no_update, dash.no_update, dash.no_update
        else:
            logs.append(html.Div("Parsing data! This might take a while ..."))
            parse_data()
            logs.append(html.Div("Step 1 complete! You can now visualize time series data and "
                                 "complete steps 2 & 3 of the setup (if not done already)."))
            global metadata
            metadata = pd.read_pickle('parsed_data/metadata.pkl')
            return "completed", "requirements met", "requirements met"

    # calculate numerical features
    @app.callback([Output(component_id='step2_wrapper', component_property='className')],
                  [Input(component_id='button_step2', component_property='n_clicks')])
    def setup_step2(_):
        if dash.callback_context.triggered[0]['prop_id'].split('.')[0] != 'button_step2':
            return dash.no_update
        if not exists('parsed_data/metadata.pkl') or len(os.listdir("parsed_data/timeseries/")) == 0:
            logs.append(html.Div('Please complete step 1 of the setup first!',
                                 style={'color': 'red', 'font-weight': 'bold'}))
            return dash.no_update
        else:
            logs.append(html.Div("Calculating features! This might take a while ..."))
            calculate_features()
            logs.append(html.Div("Step 2 complete! "
                                 "You can now visualize numerical features and use them while clustering."))
            global metadata
            metadata = pd.read_pickle('parsed_data/metadata.pkl')
            return "completed"

    # preprocess time series data
    @app.callback([Output(component_id='step3_wrapper', component_property='className'),
                   Output(component_id='step3_num_files', component_property='children')],
                  [Input(component_id='button_step3', component_property='n_clicks'),
                   Input(component_id='step3_feature', component_property='value'),
                   Input(component_id='step_3_enable_pca', component_property='value'),
                   Input(component_id='step3_PCA', component_property='value'),
                   Input(component_id='step_3_enable_rolling', component_property='value'),
                   Input(component_id='step3_rolling', component_property='value'),
                   Input(component_id='step_3_enable_weeks', component_property='value')])
    def setup_step3(_, feature, enable_pca, pca, enable_rolling, rolling, split_into_weeks):
        pca, rolling = pca[0], rolling[0]
        # check if the button was pressed or if the callback was triggered by any other input
        if dash.callback_context.triggered[0]['prop_id'].split('.')[0] != 'button_step3':
            return dash.no_update, dash.no_update
        if not exists("parsed_data/timeseries/") or len(os.listdir("parsed_data/timeseries/")) == 0:
            logs.append(html.Div('Please complete step 1 of the setup first!',
                                 style={'color': 'red', 'font-weight': 'bold'}))
            return dash.no_update, dash.no_update
        else:
            if feature is None:
                logs.append(html.Div('Please select a feature!'))
                return dash.no_update, dash.no_update
            num_files = len([f for f in os.listdir('preprocessed_data/timeseries')])
            logs.append(html.Div("Preprocessing time series data! This might take a while ..."))
            if split_into_weeks == ['yes']:
                logs.append(html.Div("Splitting data into weeks ..."))
                prepare_timeseries_per_week(feature, pca_dims=pca if enable_pca == ["enabled"] else None,
                                            window=rolling if enable_rolling == ["enabled"] else None)
            else:
                prepare_timeseries(feature, pca_dims=pca if enable_pca == ["enabled"] else None,
                                   window=rolling if enable_rolling == ["enabled"] else None)
            logs.append(html.Div("Step 3 complete! You can now use time series data while clustering."))
            return "completed", f"(Currently there are {(num_files + 1)} already preprocessed files.)"

    # create histogram / heatmap of  one / two numerical features
    @app.callback([Output(component_id='histogram', component_property='figure'),
                   Output(component_id='histogram_wrapper', component_property='className')],
                  [Input(component_id='dropdown_histogram1', component_property='value'),
                   Input(component_id='slider_histogram1', component_property='value'),
                   Input(component_id='dropdown_histogram2', component_property='value'),
                   Input(component_id='slider_histogram2', component_property='value')])
    def graph_update_histogram(feature1, bins1, feature2, bins2):
        bins1, bins2 = bins1[0], bins2[0]
        if feature1 not in metadata.columns or (feature2 not in metadata.columns and feature2 != 'none'):
            logs.append(html.Div("Please complete the setup first!"))
            return dash.no_update
        # create histogram
        if feature2 == 'none':
            # check if the slider feature 2 was changed
            if dash.callback_context.triggered[0]['prop_id'].split('.')[0] == 'slider_histogram2':
                return dash.no_update, dash.no_update
            x = metadata[feature1].to_list()
            fig = px.histogram(x, nbins=bins1, color_discrete_sequence=["#0075ff"])
            fig.update_layout(title=f'Histogram of the {get_label(feature1)}', xaxis_title=get_label(feature1),
                              yaxis_title='count', showlegend=False, height=450, width=None, title_x=0.5, autosize=True)
            return fig, "histogram"
        # create heatmap
        else:
            x = metadata[feature1].to_list()
            y = metadata[feature2].to_list()
            fig = px.density_heatmap(x=x, y=y, marginal_x="histogram", marginal_y="histogram", nbinsx=bins1,
                                     nbinsy=bins2)
            fig.update_layout(title=f'Heatmap of the {get_label(feature1)} & the {get_label(feature2)} ',
                              xaxis_title=get_label(feature1), yaxis_title=get_label(feature2),
                              showlegend=True, height=900, width=1000, title_x=0.37)
            return fig, "heatmap"

    # create barchart of a categorical feature
    @app.callback(Output(component_id='barchart', component_property='figure'),
                  [Input(component_id='dropdown_barchart', component_property='value')])
    def graph_update_barchart(feature):
        if feature not in metadata.columns:
            logs.append(html.Div("Please complete the setup first!"))
            return dash.no_update
        feat = metadata[feature].to_list()
        x, y = np.unique(np.array(feat)), []
        for i in x:
            y.append(feat.count(i))
        fig = go.Figure([go.Bar(x=x, y=y, marker={'color': ["#0075ff"] * len(x)})])
        fig.update_layout(title='Barchart', xaxis_title=get_label(feature),
                          yaxis_title='count', showlegend=False, title_x=0.5)
        return fig

    # create a graph of the motility period over time with rolling mean
    @app.callback(Output(component_id='timeseries_rolling', component_property='figure'),
                  [Input(component_id='dropdown_timeseries_rolling_id', component_property='value'),
                   Input(component_id='dropdown_timeseries_rolling_feature', component_property='value'),
                   Input(component_id='timeseries_rolling_enabled', component_property='value'),
                   Input(component_id='slider_timeseries_rolling_window', component_property='value')])
    def update_graph_timeseries_rolling(animal_id, feature, window_enabled, window):
        window = window[0]
        if dash.callback_context.triggered[0]['prop_id'].split('.')[
            0] == 'slider_timeseries_rolling_window' and window_enabled != ["enabled"]:
            return dash.no_update
        if not exists(f"parsed_data/timeseries/{animal_id}.pkl"):
            logs.append(html.Div(
                'No time series data can be displayed! Either place the parsed data into the "parsed_data" folder '
                'or place the original data in the "original_data" folder and then complete step 1 of the setup',
                style={'color': 'red', 'font-weight': 'bold'}))
            return dash.no_update
        if animal_id in already_loaded:
            df = pd.DataFrame.from_dict(already_loaded[animal_id])
            logs.append(html.Div('Loaded the time series from memory! Creating Graph ...'))
        else:
            df = pd.read_pickle(f"parsed_data/timeseries/{animal_id}.pkl")
            already_loaded[animal_id] = df.to_dict()
            logs.append(html.Div('Loaded the time series from file! Creating Graph ...'))
        df.dropna()
        if window_enabled == ['enabled']:
            df[feature] = df[feature].rolling(window, min_periods=1).mean()
        fig = go.Figure([go.Scatter(x=df['ts'], y=df[feature], line={'color': '#0075ff'})])
        fig.update_layout(title=('Rolling Mean - ' if window is not None else "") + f'{get_label(feature)} over Time',
                          xaxis_title='Time', yaxis_title=get_label(feature), title_x=0.5)
        logs.append(html.Div('Created Figure! Displaying ...'))
        return fig

    # cluster data using numerical features and time series data,
    # display in various dimensions and highlight categorical features
    @app.callback([Output(component_id='clustering', component_property='figure'),
                   Output(component_id='clustering_wrapper', component_property='style')],
                  [Input(component_id='clustering_features_numerical', component_property='value'),
                   Input(component_id='clustering_features_timeseries', component_property='value'),
                   Input(component_id='clustering_highlight_categorical', component_property='value'),
                   Input(component_id='clustering_clusters', component_property='value'),
                   Input(component_id='clustering_dimensions', component_property='value'),
                   Input(component_id='clustering_dimensions_method', component_property='value'),
                   Input(component_id='cluster_button', component_property='n_clicks'),
                   ])
    def update_graph_clustering(feat_num, feat_series, highlight_cat, num_clusters, display_dims, reduce_method,
                                button):
        feat_num, feat_series = feat_num or [], None if feat_series == 'none' else feat_series
        num_clusters = num_clusters[0]
        # check if the button was pressed or if the callback was triggered by any other input
        if dash.callback_context.triggered[0]['prop_id'].split('.')[0] != 'cluster_button':
            return dash.no_update, dash.no_update

        if len(feat_num) < 1 and feat_series is None:
            logs.append(html.Div('Please choose at least 1 feature ...'))
            return dash.no_update, dash.no_update

        if not all([f in metadata.columns for f in feat_num]):
            logs.append(html.Div("Please complete the setup first!"))
            return dash.no_update, dash.no_update

        logs.append(html.Div(f'Creating dataset and clustering...'))
        # create the dataset and cluster the data (util.py)
        cluster_dataset, assigned_clusters, axes, cluster_evals, names, used_metadata = calculate_clusters(
            metadata, feat_num, feat_series, num_clusters, highlight_cat)

        dimensions = cluster_dataset.shape[1]

        # if the dataset dimensions is larger than the wanted display dimension, reduce the dimensionality using t-SNE
        if dimensions > display_dims:
            logs.append(html.Div(f'Clustering complete! Using t-SNE to reduce dimensionality to display ...'))
            axes = [reduce_method + str(i) for i in range(1, display_dims + 1)]
            if reduce_method == 'pca':
                reduced_data = PCA(n_components=display_dims).fit_transform(cluster_dataset)
            else:
                reduced_data = TSNE(n_components=display_dims, init='random').fit_transform(cluster_dataset)
        else:
            logs.append(html.Div(f'Clustering complete! Preparing results to be displayed ...'))
            reduced_data = cluster_dataset

        c = ["Cluster #" + str(x) for x in assigned_clusters]

        if highlight_cat == "none":
            symbols = None
            cat_res_text = ""
        else:
            symbols = used_metadata[highlight_cat].to_list()
            cat_res_text = f"Comparing clusters to the highlighted category - " \
                           f"Adjusted Rand Score: {cluster_evals[highlight_cat]['rand']:0.4f}, " \
                           f"Adjusted Mutual Information Score: {cluster_evals[highlight_cat]['mis']:0.4f}"

        if dimensions < display_dims:
            logs.append(html.Div(f'You chose only {dimensions} features, but want to display the result in '
                                 f'{display_dims} dimensions! Displayed in {dimensions} dimensions ...'))
            display_dims = dimensions

        fig = display_various_dimensions(display_dims, axes, symbols, reduced_data, c, names)

        logs.append(html.Div(f'Graph created! Displaying ...'))
        fig.update_layout(
            title=f'Clustering of '
                  f'{", ".join(([get_label(f) for f in feat_num] + ([] if feat_series is None else [get_file_info(feat_series, label=True)])))} - '
                  f'Silhouette Score: {cluster_evals["sil"]:0.4f} <br> {cat_res_text}',
            legend_traceorder='normal', title_x=0.5)

        return fig, {'display': 'block'}

    def display_various_dimensions(display_dims, axes, symbols, data, colors, names):
        data = np.array(data)
        if display_dims == 1:
            fig = px.scatter(pd.DataFrame(names, columns=['Animal']), x=data[:, 0], y=[0] * data.shape[0],
                             labels={'x': get_label(axes[0]), 'y': '', 'color': 'Cluster'}, color=colors,
                             symbol=symbols, hover_data=['Animal'])
            fig.update_traces(marker={'size': 8})
            fig.update_yaxes(showgrid=False, zeroline=True, zerolinecolor='black', showticklabels=False)
            fig.update_layout(plot_bgcolor='white')
        elif display_dims == 2:
            fig = px.scatter(pd.DataFrame(names, columns=['Animal']), x=data[:, 0], y=data[:, 1], color=colors,
                             symbol=symbols,
                             hover_data=['Animal'],
                             labels={'x': get_label(axes[0]), 'y': get_label(axes[1]), 'color': 'Cluster'})
        else:
            fig = px.scatter_3d(pd.DataFrame(names, columns=['Animal']), x=data[:, 0], y=data[:, 1], z=data[:, 2],
                                color=colors,
                                hover_data=['Animal'], symbol=symbols,
                                labels={'x': get_label(axes[0]), 'y': get_label(axes[1]), 'z': get_label(axes[2]),
                                        'color': 'Cluster'})
        return fig

    # create a histogram for a feature of a single animal
    @app.callback(Output(component_id='clustering_single_histogram', component_property='figure'),
                  [Input(component_id='dropdown_clustering_histogram_id', component_property='value'),
                   Input(component_id='dropdown_clustering_histogram_feature', component_property='value'),
                   Input(component_id='slider_clustering_histogram_bins', component_property='value'),
                   Input(component_id='clustering_histogram_enable_rolling', component_property='value'),
                   Input(component_id='slider_clustering_histogram_rolling_window', component_property='value')])
    def graph_update_clustering_single_histogram(animal_id, feature, bins, enable_window, window):
        window, bins = window[0], bins[0]
        if dash.callback_context.triggered[0]['prop_id'].split('.')[
            0] == 'slider_clustering_histogram_rolling_window' and enable_window != ["enabled"]:
            return dash.no_update
        if not exists(f"parsed_data/timeseries/{animal_id}.pkl"):
            logs.append(html.Div(
                'No time series data can be displayed! Either place the parsed data into the "parsed_data" folder '
                'or place the original data in the "original_data" folder and then complete step 1 of the setup',
                style={'color': 'red', 'font-weight': 'bold'}))
            return dash.no_update
        if animal_id in already_loaded:
            df = pd.DataFrame.from_dict(already_loaded[animal_id])
            logs.append(html.Div('Loaded the time series from memory! Creating Graph ...'))
        else:
            df = pd.read_pickle(f"parsed_data/timeseries/{animal_id}.pkl")
            already_loaded[animal_id] = df.to_dict()
            logs.append(html.Div('Loaded the time series from file! Creating Graph ...'))
        df.dropna()
        if enable_window == ["enabled"]:
            df[feature] = df[feature].rolling(window, min_periods=1).mean()

        x = df[feature].to_list()
        if feature == 'mot_period':
            min_data = metadata.loc[[metadata['min'].idxmax()]]['min'].values[0]
            max_data = metadata.loc[[metadata['max'].idxmax()]]['max'].values[0]
        elif feature == 'mot_pulse_width':
            min_data, max_data = 4, 17
        else:
            min_data, max_data = 0, 1
        # max value animal id - '60ec0f2df9fd09de9dcd67e0'
        fig = px.histogram(x, nbins=bins, range_x=[min_data, max_data],
                           color_discrete_sequence=["#0075ff"], histnorm='probability density')
        fig.update_layout(title=f'Histogram of the {get_label(feature)}' +
                                ('- Rolling Mean - ' if enable_window == ["enabled"] else ""),
                          xaxis_title=get_label(feature), yaxis_title='percentage', showlegend=False, title_x=0.5)
        logs.append(html.Div('Created Figure! Displaying ...'))
        return fig

    # cluster data using numerical features and time series data,
    # display in various dimensions and highlight categorical features
    @app.callback(Output(component_id='preprocessed_histograms', component_property='children'),
                  [Input(component_id='dropdown_clustering_histogram_preprocess_feature', component_property='value'),
                   Input(component_id='clustering_histogram_preprocess_enable_rolling', component_property='value'),
                   Input(component_id='slider_clustering_histogram_preprocess_rolling_window',
                         component_property='value'),
                   Input(component_id='slider_clustering_histogram_preprocess_bins', component_property='value'),
                   Input(component_id='clustering_histogram_preprocess_enable_weeks', component_property='value'),
                   Input(component_id='preprocess_histogram_button', component_property='n_clicks')
                   ])
    def preprocess_histogram_callback(feature, enable_window, window, bins, split_into_weeks, button):
        # check if the button was pressed or if the callback was triggered by any other input
        window, bins = window[0], bins[0]
        if dash.callback_context.triggered[0]['prop_id'].split('.')[0] != 'preprocess_histogram_button':
            return dash.no_update

        logs.append(html.Div(f'Preprocessing histograms! This might take a while...'))
        if split_into_weeks:
            preprocess_histograms_per_week(feature, enable_window, window, bins)
        else:
            preprocess_histograms(feature, enable_window, window, bins)

        num_files = len([f for f in os.listdir('preprocessed_data/histograms')])
        return f"(There are {(num_files + 1)} different versions of preprocessed histograms)"

    # cluster histograms using the preprocessed files
    @app.callback([Output(component_id='clustering_histogram', component_property='figure'),
                   Output(component_id='clustering_histogram_wrapper', component_property='style')],
                  [Input(component_id='clustering_histogram_file', component_property='value'),
                   Input(component_id='clustering_histogram_method', component_property='value'),
                   Input(component_id='slider_clusters_histogram', component_property='value'),
                   Input(component_id='clustering_histogram_dimensions', component_property='value'),
                   Input(component_id='clustering_histogram_dimensions_method', component_property='value'),
                   Input(component_id='clustering_histogram_highlight_categorical', component_property='value'),
                   Input(component_id='clustering_histogram_button', component_property='n_clicks')
                   ])
    def graph_update_histogram_clustering(file, method, clusters, display_dimensions, reduce_method, highlight_cat,
                                          button):
        clusters = clusters[0]

        # check if the button was pressed or if the callback was triggered by any other input
        if dash.callback_context.triggered[0]['prop_id'].split('.')[0] != 'clustering_histogram_button':
            return dash.no_update, dash.no_update

        _, _, _, split_into_weeks = get_file_info(file)
        if method == 'kmeans_cos' and split_into_weeks:
            logs.append(html.Div(f'This method is too resource-intensive when the data is split into weeks. '
                                 f'Please select another file or method!'))
            return dash.no_update, dash.no_update

        method_description = {
            'kmeans': 'K-Means',
            'kmeans_cos': 'Cosine similarity of every pair',
            'kmeans_avg_cos': 'Cosine similarity to the average',
            'kmeans_avg_kl': 'Kullback–Leibler divergence to the average',
        }

        if method == 'kmeans_avg_cos' or method == 'kmeans_avg_kl':
            display_dimensions = 1
            axes = [method]
        else:
            axes = [reduce_method + str(i) for i in range(1, display_dimensions + 1)]

        cluster_dataset, assigned_labels, names, cluster_evals, used_metadata = cluster_histograms(metadata, file,
                                                                                                   method, clusters,
                                                                                                   highlight_cat)

        dimensions = np.array(cluster_dataset).shape[1]

        if dimensions <= display_dimensions:
            reduced_data = cluster_dataset
            display_dimensions = dimensions
        elif reduce_method == 'pca':
            reduced_data = PCA(n_components=display_dimensions).fit_transform(cluster_dataset)
        elif reduce_method == 'tsne':
            reduced_data = TSNE(n_components=display_dimensions, init='random').fit_transform(cluster_dataset)
        else:
            reduced_data = cluster_dataset

        c = ["Cluster #" + str(x) for x in assigned_labels]

        if highlight_cat == "none":
            symbols = None
            cat_res_text = ""
        else:
            symbols = used_metadata[highlight_cat].to_list()
            cat_res_text = f"Comparing clusters to the highlighted category - " \
                           f"Adjusted Rand Score: {cluster_evals[highlight_cat]['rand']:0.4f}, " \
                           f"Adjusted Mutual Information Score: {cluster_evals[highlight_cat]['mis']:0.4f}"

        fig = display_various_dimensions(display_dimensions, axes, symbols, reduced_data, c, names)
        logs.append(html.Div(f'Graph created! Displaying ...'))
        fig.update_layout(
            title=f'Clustering of histograms - {method_description[method]} -'
                  f'Silhouette Score: {cluster_evals["sil"]:0.4f} <br> {cat_res_text}',
            legend_traceorder='normal', title_x=0.5)
        return fig, {'display': 'block'}

    # print the best results to the log
    @app.callback([
        Output(component_id='results_textarea', component_property='value'),
        Output(component_id='results_textarea', component_property='style')
    ], [Input(component_id='button_results', component_property='n_clicks'),
        Input(component_id='top_n_results', component_property='value')])
    def display_best_results(button, top_n_results):
        # check if the button was pressed or if the callback was triggered by any other input
        if dash.callback_context.triggered[0]['prop_id'].split('.')[0] != 'button_results':
            return dash.no_update, dash.no_update
        return [print_best_clusters(top_n_results[0], features=features).replace('[4m', '')
                    .replace('[1m', '').replace('[0m', '')], {'display': 'block'}

    if __name__ == '__main__':
        app.run_server()


def main():
    start_dashboard()


if __name__ == '__main__':
    main()
