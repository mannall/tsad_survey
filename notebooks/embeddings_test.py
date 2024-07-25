import numpy as np
import pandas as pd
import json

import dash
import dash_bootstrap_components as dbc
import plotly.express as px

from pathlib import Path

# Read the embeddings data
embeddings = pd.read_csv(Path('embeddings.csv'))

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dash.html.H1("t-SNE Visualization"),
        dash.html.Hr(),
        dbc.Row(
            [
                dbc.Col(dash.dcc.Graph(id="tsne-graph"), md=5),
                dbc.Col(dash.dcc.Graph(id="ts-graph"), md=5),
                dbc.Col(dash.html.P(id='click-data'), md=2)
            ],
            # align="center",
        ),
    ],
    fluid=True,
)

@app.callback(
    dash.Output('tsne-graph', 'figure'),
    [dash.Input('tsne-graph', 'id')]
)
def update_graph(input_value):    
    fig = px.scatter(embeddings, x='TSNE1', y='TSNE2', color='Dataset', 
                     hover_data=['Path'], color_continuous_scale='Spectral')
    fig.update_traces(marker=dict(size=8, opacity=0.75))
    fig.update_layout(coloraxis_colorbar=dict(title='Dataset'))
    return fig

@app.callback(
    dash.Output('click-data', 'children'),
    dash.Input('tsne-graph', 'clickData')
)
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)

@app.callback(
    dash.Output('ts-graph', 'figure'),
    dash.Input('tsne-graph', 'clickData')
)
def display_ts_data(clickData):
    if clickData is not None: y, _ = np.hsplit(pd.read_csv(clickData['points'][0]['customdata'][0]).values, 2)
    fig = px.line(x=range(min(len(y.flatten()), 2500)), y=y.flatten()[:2500])
    return fig

if __name__ == "__main__":
    app.run_server()




# app.layout = html.Div([


#     html.H1(children='t-SNE Visualization', style={'textAlign': 'center'}),
#     dcc.Dropdown(embeddings['Dataset'].unique(), embeddings['Dataset'].unique()[0], id='dataset-dropdown'),
#     dcc.Graph(id='tsne-graph')
# ])

# @callback(
#     Output('tsne-graph', 'figure'),
#     Input('dataset-dropdown', 'value')
# )
# def update_graph(selected_dataset):
#     if selected_dataset:
#         dff = embeddings[embeddings['Dataset'] == selected_dataset]
#     else:
#         dff = embeddings
    
#     fig = px.scatter(dff, x='TSNE1', y='TSNE2', color='Dataset', 
#                      hover_data=['File'], color_continuous_scale='Spectral', 
#                      title='t-SNE Visualization')
#     fig.update_traces(marker=dict(size=8, opacity=0.75))
#     fig.update_layout(coloraxis_colorbar=dict(title='Dataset'))
    
#     return fig

# if __name__ == '__main__':
#     app.run(debug=True)