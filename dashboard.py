import numpy as np
import pandas as pd
import dash
from dash import html
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc
from dash.dependencies import Input, Output
from sklearn.decomposition import PCA
import os
import flask
from os.path import exists
from util import calculate_clusters, get_file_info, parse_data, calculate_features, prepare_timeseries

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
    ['iqr', 'Interquartile range', 'numerical'],
    ['count_nan', 'Percentage of invalid measurements', 'numerical'],
    ['group_id', 'Group ID of an animal', 'categorical'],
    ['organisation_id', 'Organisation ID of an animal', 'categorical'],
    ['organisation_timezone', 'Timezone the animal is located in', 'categorical'],
    ['pca1', "Principal Component 1", 'pca'],
    ['pca2', "Principal Component 2", 'pca'],
    ['pca3', "Principal Component 3", 'pca'],
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
    preprocessed_data_options = [{'label': "No time series", 'value': 'none'}]
    preprocessed_files = os.listdir("preprocessed_data/")
    for f in preprocessed_files:
        lab = get_file_info(f, label=True)
        preprocessed_data_options.append({'label': lab, 'value': f})

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
            html.Div(id='step3_wrapper', className='completed' if len(preprocessed_files) else '', children=[
                html.H3("Step 3: Preprocessing time series data"),
                html.Span(" - already completed!"),
                html.P(
                    "The time series data is not of the same length, not normalized and usually too big to use for "
                    "clustering. Here, the data will be preprocessed and stored.", className='setup_explanation'),
                html.P(f"(Currently there are {len(preprocessed_files)} already preprocessed files.)",
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
                              dcc.Slider(id="step3_PCA", min=2, max=2000, value=10, marks={2: '2', 2000: '2000'}),
                              html.Span('Rolling mean window:'),
                              dcc.Checklist(id='step_3_enable_rolling',
                                            options=[{'label': 'enable', 'value': 'enabled'}]),
                              dcc.Slider(id="step3_rolling", min=2, max=1000, value=10, marks={2: '2', 1000: '1000'})]),
                    html.Span("Enables: ", className='enables_span'),
                    html.Span("Using time series data for clustering"),
                    html.Button("Go!", id="button_step3"),
                ])])
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
            dcc.Slider(id="slider_histogram1", min=2, max=100, value=50, marks={2: '2', 100: '100'})
        ]),
        html.Div(className="half", children=[
            html.H2("Feature 2:"),
            dcc.Dropdown(id='dropdown_histogram2', value='none',
                         options=[{'label': 'No Feature', 'value': 'none'}] +
                                 features[features['category'] == 'numerical'][['label', 'value']].to_dict('records')),
            html.P("Number of Bins:"),
            dcc.Slider(id="slider_histogram2", min=2, max=100, value=50, marks={2: '2', 100: '100'})
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

    # Create time series tab layout
    layouts['layout_timeseries'] = html.Div(id='parent_timeseries', children=[
        html.H1(children='Visualizing Time Series Data'),
        html.P("Animal ID:"),
        dcc.Dropdown(id='dropdown_timeseries_id', options=animal_id_options, value=animal_id_options[0]['value']),
        html.P("Feature to visualize:"),
        dcc.Dropdown(id='dropdown_timeseries_feature', value='mot_period',
                     options=features[features['category'] == 'time'][['label', 'value']].to_dict('records')),
        dcc.Loading(id="loading-icon-3", type="default", children=[html.Div(dcc.Graph(id='timeseries'))])
    ])

    # Create time series with rolling mean tab layout
    layouts['layout_timeseries_rolling'] = html.Div(id='parent_timeseries_rolling', children=[
        html.H1(children='Visualizing Time Series Data - Rolling Mean'),
        html.P("Animal ID:"),
        dcc.Dropdown(id='dropdown_timeseries_rolling_id', options=animal_id_options,
                     value=animal_id_options[0]['value']),
        html.P("Feature to visualize:"),
        dcc.Dropdown(id='dropdown_timeseries_rolling_feature', value='mot_period',
                     options=features[features['category'] == 'time'][['label', 'value']].to_dict('records')),
        html.P("Size of Window:"),
        dcc.Slider(id="slider_timeseries_rolling_window", min=2, max=1000, value=30, marks={2: '2', 1000: '1000'}),
        dcc.Loading(id="loading-icon-4", type="default", children=[html.Div(dcc.Graph(id='timeseries_rolling'))])
    ])

    # Create clustering tab layout
    layouts['layout_clustering'] = html.Div(id='parent_clustering', children=[
        html.H1(children='Clustering'),
        html.Div(className="wrapper", children=[
            html.H2("Numerical Features:", style={'font-weight': 'bold', 'margin-top': '30px'}),
            dcc.Checklist(id='clustering_features_numerical', value=['mean', 'std'],
                          options=features[features['category'] == 'numerical'][['label', 'value']].to_dict('records')),
            html.H2("Time Series Data:"),
            dcc.Dropdown(id='clustering_features_timeseries', options=preprocessed_data_options, value='none'),
            html.H2("Category to Highlight:"),
            dcc.RadioItems(id='clustering_highlight_categorical', value='none',
                           options=[{'label': 'No Feature', 'value': 'none'}] +
                                   features[features['category'] == 'categorical'][['label', 'value']].to_dict(
                                       'records')),
            html.Div(className="half", children=[
                html.H2("Number of Clusters:"),
                dcc.Slider(id="clustering_clusters", min=2, max=15, value=3, marks={2: '2', 15: '15'})]),
            html.Div(className="half", children=[
                html.H2("Display in:"),
                dcc.RadioItems(id='clustering_dimensions', value=2,
                               options=[{'label': '1 Dimension', 'value': 1}, {'label': '2 Dimensions', 'value': 2},
                                        {'label': '3 Dimensions', 'value': 3}]),

            ]),
            html.Button('Calculate Clusters!', id='cluster_button'),
        ]),
        dcc.Loading(id="loading-icon-5", type="default",
                    children=[html.Div(id='clustering_wrapper', children=[dcc.Graph(id='clustering')])])
    ])

    # Create layout of the entire webapp
    app.layout = html.Div([
        html.Link(href='/static/dashboard.css', rel='stylesheet'),
        dcc.Tabs(id="tabs", children=[
            dcc.Tab(label='Setup', value='layout_setup'),
            dcc.Tab(label='Feature Histogram', value='layout_histogram'),
            dcc.Tab(label='Feature Barchart', value='layout_barchart'),
            dcc.Tab(label='Timeseries', value='layout_timeseries'),
            dcc.Tab(label='Timeseries - Rolling Mean', value='layout_timeseries_rolling'),
            dcc.Tab(label='Clustering', value='layout_clustering')
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
                   Input(component_id='step3_rolling', component_property='value')])
    def setup_step3(_, feature, enable_pca, pca, enable_rolling, rolling):
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
            num_files = len([f for f in os.listdir('preprocessed_data/')])
            logs.append(html.Div("Preprocessing time series data! This might take a while ..."))
            prepare_timeseries(feature, pca_dims=pca if enable_pca == ["enabled"] else None,
                               window=rolling if enable_rolling == ["enabled"] else None)
            logs.append(html.Div("Step 3 complete! You can now use time series data while clustering."))
            return "completed", f"(Currently there are {(num_files+1)} already preprocessed files.)"

    # create histogram / heatmap of  one / two numerical features
    @app.callback([Output(component_id='histogram', component_property='figure'),
                   Output(component_id='histogram_wrapper', component_property='className')],
                  [Input(component_id='dropdown_histogram1', component_property='value'),
                   Input(component_id='slider_histogram1', component_property='value'),
                   Input(component_id='dropdown_histogram2', component_property='value'),
                   Input(component_id='slider_histogram2', component_property='value')])
    def graph_update_histogram(feature1, bins1, feature2, bins2):
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

    # create a graph of any feature over time with and without rolling mean
    def generate_timeseries_graph(animal_id, feature, rolling=None):
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
        if rolling is not None:
            df[feature] = df[feature].rolling(rolling, min_periods=1).mean()
        fig = go.Figure([go.Scatter(x=df['ts'], y=df[feature], line={'color': '#0075ff'})])
        fig.update_layout(title=('Rolling Mean - ' if rolling is not None else "") + f'{get_label(feature)} over Time',
                          xaxis_title='Time', yaxis_title=get_label(feature), title_x=0.5)
        logs.append(html.Div('Created Figure! Displaying ...'))
        return fig

    # create a graph of the motility period over time
    @app.callback(Output(component_id='timeseries', component_property='figure'),
                  [Input(component_id='dropdown_timeseries_id', component_property='value'),
                   Input(component_id='dropdown_timeseries_feature', component_property='value')])
    def graph_update_timeseries(animal_id, feature):
        return generate_timeseries_graph(animal_id, feature)

    # create a graph of the motility period over time with rolling mean
    @app.callback(Output(component_id='timeseries_rolling', component_property='figure'),
                  [Input(component_id='dropdown_timeseries_rolling_id', component_property='value'),
                   Input(component_id='dropdown_timeseries_rolling_feature', component_property='value'),
                   Input(component_id='slider_timeseries_rolling_window', component_property='value')])
    def graph_update_graph_timeseries_rolling(animal_id, feature, window):
        return generate_timeseries_graph(animal_id, feature, rolling=window)

    # cluster data using numerical features and time series data,
    # display in various dimensions and highlight categorical features
    @app.callback([Output(component_id='clustering', component_property='figure'),
                   Output(component_id='clustering_wrapper', component_property='style')],
                  [Input(component_id='clustering_features_numerical', component_property='value'),
                   Input(component_id='clustering_features_timeseries', component_property='value'),
                   Input(component_id='clustering_highlight_categorical', component_property='value'),
                   Input(component_id='clustering_clusters', component_property='value'),
                   Input(component_id='clustering_dimensions', component_property='value'),
                   Input(component_id='cluster_button', component_property='n_clicks'),
                   ])
    def graph_update_graph_clustering(feat_num, feat_series, highlight_cat, num_clusters, display_dims, button):
        feat_num, feat_series = feat_num or [], None if feat_series == 'none' else feat_series

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
        cluster_dataset, assigned_clusters, axes, cluster_evals = calculate_clusters(
            metadata, feat_num, feat_series, num_clusters)

        dimensions = cluster_dataset.shape[1]

        # if the dataset dimensions is larger than the wanted display dimension, reduce the dimensionality using PCA
        if dimensions > display_dims:
            logs.append(html.Div(f'Clustering complete! Using PCA to reduce dimensionality to display ...'))
            axes = ['pca' + str(i) for i in range(1, display_dims + 1)]
            pca = PCA(n_components=display_dims)
            pca_data = pca.fit_transform(cluster_dataset)
        else:
            logs.append(html.Div(f'Clustering complete! Preparing results to be displayed ...'))
            pca_data = cluster_dataset

        c = ["Cluster #" + str(x) for x in assigned_clusters]

        if highlight_cat == "none":
            symbols = None
            cat_res_text = ""
        else:
            symbols = metadata[highlight_cat].to_list()
            cat_res_text = f"Comparing clusters to the highlighted category - " \
                           f"Adjusted Rand Score: {cluster_evals[highlight_cat]['rand']:0.4f}, " \
                           f"Adjusted Mutual Information Score: {cluster_evals[highlight_cat]['mis']:0.4f}"

        if dimensions < display_dims:
            logs.append(html.Div(f'You chose only {dimensions} features, but want to display the result in '
                                 f'{display_dims} dimensions! Displayed in {dimensions} dimensions ...'))
            display_dims = dimensions

        if display_dims == 1:
            fig = px.scatter(x=pca_data[:, 0], y=[0] * cluster_dataset.shape[0],
                             labels={'x': get_label(axes[0]), 'y': '', 'color': 'Cluster'}, color=c, symbol=symbols)
            fig.update_traces(marker={'size': 12})
            fig.update_yaxes(showgrid=False, zeroline=True, zerolinecolor='black', showticklabels=False)
            fig.update_layout(plot_bgcolor='white')
        elif display_dims == 2:
            fig = px.scatter(x=pca_data[:, 0], y=pca_data[:, 1], color=c, symbol=symbols,
                             labels={'x': get_label(axes[0]), 'y': get_label(axes[1]), 'color': 'Cluster'})
        else:
            fig = px.scatter_3d(x=pca_data[:, 0], y=pca_data[:, 1], z=pca_data[:, 2], color=c, symbol=symbols,
                                labels={'x': get_label(axes[0]), 'y': get_label(axes[1]), 'z': get_label(axes[2]),
                                        'color': 'Cluster'})
        logs.append(html.Div(f'Graph created! Displaying ...'))
        fig.update_layout(
            title=f'Clustering of '
                  f'{", ".join(([get_label(f) for f in feat_num] + ([] if feat_series is None else [get_file_info(feat_series, label=True)])))} - '
                  f'Silhouette Score: {cluster_evals["sil"]:0.4f} <br> {cat_res_text}',
            legend_traceorder='normal', title_x=0.5)

        return fig, {'display': 'block'}

    if __name__ == '__main__':
        app.run_server()


def main():
    start_dashboard()


if __name__ == '__main__':
    main()
