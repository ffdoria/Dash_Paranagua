import os

import dash
import dash_bootstrap_components  as dbc
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error

def fill_small_gaps_corrected(df, threshold=12):
    threshold *= 3
    if isinstance(df.index[0], str):
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
        
    # Reindex the dataframe to include all expected datetime indices
    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='20T')
    df_reindexed = df.reindex(all_dates)
    
    # Identify where the gaps are
    missing_data = df_reindexed[df.columns[0]].isnull().astype(int)
    # Compute a "grouping" series that identifies consecutive missing data
    groups = (missing_data.diff() != 0).cumsum()
    
    # Count the size of each group
    gap_sizes = groups.value_counts()

    # Identify groups that are small gaps (less than or equal to the threshold)
    small_gaps = gap_sizes[gap_sizes <= threshold].index
    
    # Create a mask for where to interpolate
    interpolate_mask = groups.isin(small_gaps) & missing_data
    
    # Interpolate only for the small gaps
    interpolated_data = df_reindexed[df.columns[0]].interpolate()
    df_reindexed[df.columns[0]] = np.where(interpolate_mask, interpolated_data, df_reindexed[df.columns[0]])
    
    return df_reindexed
def filter_dataframe(df_real,df_predict, target_date_str,dias_historico=5,separar=True):
    pontos_tras = dias_historico*3*24
    pontos_frente = 24*3
    
    # Converta o índice para datetime se ainda não for
    if not isinstance(df_real.index, pd.DatetimeIndex):
        df_real.index = pd.to_datetime(df_real.index, format='%d/%m/%Y %H:%M')
    
    if not isinstance(df_predict.index, pd.DatetimeIndex):
        df_predict.index = pd.to_datetime(df_predict.index, format='%d/%m/%Y %H:%M')
    
    # Localize o índice da linha para a data fornecida
    target_date_real = pd.to_datetime(target_date_str, format='%d/%m/%Y %H:%M')
    target_idx_real = df_real.index.get_loc(target_date_real)
    
    target_date_predict = pd.to_datetime(target_date_str, format='%d/%m/%Y %H:%M')
    target_idx_predict = df_predict.index.get_loc(target_date_predict)
    

    
    if not separar:
        # Se o índice foi encontrado, filtre o DataFrame para incluir essa linha
        real = df_real.iloc[target_idx_real-pontos_tras:target_idx_real+72]
        real_predict = df_real.iloc[target_idx_real:target_idx_real+72]
        return (real,real_predict,None)

    else:
        # Se o índice foi encontrado, filtre o DataFrame para incluir essa linha
        real_tras = df_real.iloc[target_idx_real-pontos_tras:target_idx_real+1]
        real_frente = df_real.iloc[target_idx_real:target_idx_real + 72]
        predict_frente = df_predict.iloc[target_idx_predict:target_idx_predict + 72]
        return (real_tras,real_frente,predict_frente)   

# path = 'C:\\Users\\ffdor\\OneDrive\\Área de Trabalho\\TCC POLI\\Colab\\TCC\\'
path = ''

predictions = pd.read_csv(path+'ALL_PREDICTIONS.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
predictions.rename(columns={'0': 'Altura'}, inplace=True)

real = pd.read_csv(path+'Erro_Galheta.CSV', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
real.rename(columns={'0': 'Altura'}, inplace=True)
real = fill_small_gaps_corrected(real, threshold=12)

ssh_real = pd.read_csv(path+'GALHETA_SSH.CSV', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
ssh_real.rename(columns={'0': 'Altura'}, inplace=True)
ssh_real = fill_small_gaps_corrected(ssh_real, threshold=12)

ssh_estimado = pd.read_csv(path+'GALHETA_SSH_PRED.CSV', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
ssh_estimado.rename(columns={'0': 'Altura'}, inplace=True)

teorico = pd.read_csv(path+'GALHETA_TEO.CSV', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
teorico.rename(columns={'0': 'Altura'}, inplace=True)


###############DASH

app_name = os.getenv("APP_NAME", "Paranagua")

TEMPLATE = "plotly_white"

app = dash.Dash(
    external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True
)
app.title = "Previsão Paranaguá"
server = app.server

"""Homepage"""
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content"),]
)

index_page = html.Div(
    [
        html.Br(),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.H1(children="Previsão do nível da maré no Porto de Paranaguá"), width=8),
                dbc.Col(width=8),
            ],
            justify="center",
        ),
        html.Br(),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H4(
                                children="A previsão é feita para as localizações Porto Cais Oeste e Galheta"
                            ),
                            html.Div(
                                [
                                    dcc.Link(
                                        html.Button(
                                            "Home", id="home-button", className="mr-1"
                                        ),
                                        href=f"/{app_name}/",
                                    ),
                                    dcc.Link(
                                        html.Button(
                                            "Cais Oeste", id="cais-oeste-button", className="mr-1"
                                        ),
                                        href=f"/{app_name}/cais-oeste",
                                    ),
                                    dcc.Link(
                                        html.Button(
                                            "Galheta", id="galheta-button", className="mr-1"
                                        ),
                                        href=f"/{app_name}/galheta",
                                    )
                                ]
                            ),
                                html.Br(),
                                    html.Br(),
                                    html.Br(),
                            html.H4(children="Para estimar o nível da maré foi utilizado LSTM.xZZZx...."),
                        ]
                    ),
                    width=7,
                ),
                dbc.Col(width=3),
            ],
            justify="center",
        ),
        html.Br(),
        html.Br(),
        html.Br(),

    ]
)



"""GALHETA LAYOUT"""
galheta_layout = html.Div(
    [
        html.Div(id="galheta-content"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dcc.Link(
                                html.Button("HOME", id="home-button", className="mr-1"),
                                href=f"/{app_name}/",
                            ),
                            dcc.Link(
                                html.Button(
                                    "Cais Oeste", id="cais-oeste-button", className="mr-1"
                                ),
                                href=f"/{app_name}/cais-oeste",
                            ),
                            dcc.Link(
                                html.Button("Galheta", id="galheta-button", className="mr-1"),
                                href=f"/{app_name}/galheta",
                            ),
                        ]
                    ),
                    width=4,
                ),
                dbc.Col(width=7),
            ],
            justify="center",
        ),
        html.Br(),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.H1("Previsão para a estação Galheta"), width=9
                ),
                dbc.Col(width=2),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        children="""Escolha a data da previsão:"""
                    ),
                    width=2,
                ),
                dbc.Col(width=9),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [   
                dbc.Col(
                    dcc.DatePickerSingle(
                        id='data-galheta',
                        date=date(2021, 3, 31),
                        min_date_allowed=date(2021, 3, 31),
                        max_date_allowed=date(2022, 8, 14),
                        display_format='DD/MM/YYYY'
                    ),width=4), 
                    dbc.Col(width=7),
            ], 
            justify="center",
        ),

        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.H4(
                        children="""Previsão da Maré Meteorológica"""
                    ),
                    width=9,
                ),
                dbc.Col(width=2),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="galheta-dropdown",
                        options=[
                            {"label": "Historico", "value": "Historico"},
                            {"label": "Futuro", "value": "Futuro"},
                        ],
                        value=["Historico", "Futuro"],
                        multi=True,
                    ),
                    width=6,
                ),
                dbc.Col(width=5),
            ],
            justify="center",
        ),
        dcc.Graph(id="galheta-graph"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id='output-erro'),
                    width=9,
                ),
                dbc.Col(width=2),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.H4(
                        children="""Previsão do SSH"""
                    ),
                    width=9,
                ),
                dbc.Col(width=2),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="galheta-dropdown-ssh",
                        options=[
                            {"label": "Maré Astronomica", "value": "Maré Astronomica"},
                            {"label": "Maré Medida", "value": "Maré Medida"},
                            {"label": "Maré Estimada", "value": "Maré Estimada"},
                        ],
                        value=["Maré Astronomica", "Maré Medida",'Maré Estimada'],
                        multi=True,
                    ),
                    width=6,
                ),
                dbc.Col(width=5),
            ],
            justify="center",
        ),
        dcc.Graph(id="galheta-graph-ssh"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id='output-erro-ssh'),
                    width=9,
                ),
                dbc.Col(width=2),
            ],
            justify="center",
        ),

        html.Br(),        
    ]
)


@app.callback(
    dash.dependencies.Output("galheta-content", "children"),
    [dash.dependencies.Input("galheta-button", "value")],
)
@app.callback(
    dash.dependencies.Output("galheta-graph", "figure"),
    [dash.dependencies.Input("galheta-dropdown", "value"), dash.dependencies.Input("data-galheta", "date")],
)
def plot_galheta(value,data_escolhida):
    fig = go.Figure()
    data = datetime.strptime(data_escolhida, '%Y-%m-%d')
    history, true_future, prediction = filter_dataframe(real, predictions, data)

    index_history =history.index
    index_target = true_future.index

    if "Historico" in value:
        fig.add_trace(
            go.Scatter(
                x=index_history,
                y=np.array(history.values)[:,0],
                name="Dados Passado",
                line=dict(color="blue", width=3),
            )
        )

    if "Futuro" in value:
        fig.add_trace(
            go.Scatter(
                x=index_target,
                y=np.array(true_future.values)[:,0],
                name="Dados Corretos",
                line=dict(color="lightblue", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_target,
                y=np.array(prediction.values)[:,0],
                name="Dados Estimados",
                line=dict(color="red", width=3, dash="dash"),
            )
        )

    return fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Altura(m)",
        template=TEMPLATE,
    )

@app.callback(
    dash.dependencies.Output('output-erro', 'children'),
    [dash.dependencies.Input("data-galheta", "date")]
)
def update_div(data):
    data = datetime.strptime(data, '%Y-%m-%d')
    history, true_future, prediction = filter_dataframe(real, predictions, data)

    MSE = mean_squared_error(true_future,prediction)
    RMSE =  np.sqrt(MSE)

    return f'O erro dessa é previsão é MSE: {round(MSE,3)} e RMSE: {round(RMSE,3)}.'


@app.callback(
    dash.dependencies.Output("galheta-graph-ssh", "figure"),
    [dash.dependencies.Input("galheta-dropdown-ssh", "value"), dash.dependencies.Input("data-galheta", "date")],
)
def plot_galheta_ssh(value,data):
    fig = go.Figure()
    data = datetime.strptime(data, '%Y-%m-%d')
    history_ssh, true_future_ssh, prediction_ssh = filter_dataframe(ssh_real, ssh_estimado, data)
    
    history_teo, _ , _ = filter_dataframe(teorico, ssh_estimado, data,separar=False)

    index_history =history_ssh.index
    index_target = true_future_ssh.index
    index_teo = history_teo.index

    if "Maré Astronomica" in value:
        fig.add_trace(
            go.Scatter(
                x=index_teo,
                y=np.array(history_teo.values)[:,0],
                name="Maré Astronômica",
                line=dict(color="blue", width=3),
            )
        )

    if "Maré Medida" in value:
        fig.add_trace(
            go.Scatter(
                x=index_history,
                y=np.array(history_ssh.values)[:,0],
                name="SSH Medido Historico",
                line=dict(color="orange", width=3),
            )
        )
    

    if "Maré Estimada" in value:
        fig.add_trace(
            go.Scatter(
                x=index_target,
                y=np.array(prediction_ssh.values)[:,0],
                name="Dados Estimados",
                line=dict(color="red", width=3, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_target,
                y=np.array(true_future_ssh.values)[:,0],
                name="SSH Medido",
                line=dict(color="orange", width=3),
            )
        )

    return fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Altura(m)",
        template=TEMPLATE,
    )
@app.callback(
    dash.dependencies.Output('output-erro-ssh', 'children'),
    [dash.dependencies.Input("data-galheta", "date")]
)
def update_div2(data):
    data = datetime.strptime(data, '%Y-%m-%d')
    history_ssh, true_future_ssh, prediction_ssh = filter_dataframe(ssh_real, ssh_estimado, data)
    _, predict_teo , _ = filter_dataframe(teorico, ssh_estimado, data,separar=False)

    MSE_teo = mean_squared_error(true_future_ssh,predict_teo)
    RMSE_teo =  np.sqrt(MSE_teo)

    return f'Erro entre o SSH e a maré Astronômica é de MSE: {round(MSE_teo,3)} e RMSE: {round(RMSE_teo,3)}.'
    

"""CAIS OESTE LAYOUT"""
cais_oeste_layout = html.Div(
    [
        html.Div(id="cais-oeste-content"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dcc.Link(
                                html.Button("HOME", id="home-button", className="mr-1"),
                                href=f"/{app_name}/",
                            ),
                            dcc.Link(
                                html.Button(
                                    "Cais Oeste", id="cais-oeste-button", className="mr-1"
                                ),
                                href=f"/{app_name}/cais-oeste",
                            ),
                            dcc.Link(
                                html.Button("Galheta", id="galheta-button", className="mr-1"),
                                href=f"/{app_name}/galheta",
                            ),
                        ]
                    ),
                    width=4,
                ),
                dbc.Col(width=7),
            ],
            justify="center",
        ),
        html.Br(),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.H1("Previsão para o porto de Cais Oeste"), width=9
                ),
                dbc.Col(width=2),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        children="""Escolha a data da previsão:"""
                    ),
                    width=2,
                ),
                dbc.Col(width=9),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [   
                dbc.Col(
                    dcc.DatePickerSingle(
                        id='data-cais-oeste',
                        date=date(2021, 3, 31),
                        min_date_allowed=date(2021, 3, 31),
                        max_date_allowed=date(2022, 8, 14),
                        placeholder='Select a date',  # Placeholder is a string
                        display_format='DD/MM/YYYY'
                    ),width=4), 
                    dbc.Col(width=7),
            ], 
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        children="""Erro estimado de ..."""
                    ),
                    width=9,
                ),
                dbc.Col(width=2),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="cais-oeste-dropdown",
                        options=[
                            {"label": "Historico", "value": "Historico"},
                            {"label": "Futuro", "value": "Futuro"},
                        ],
                        value=["Historico", "Futuro"],
                        multi=True,
                    ),
                    width=6,
                ),
                dbc.Col(width=5),
            ],
            justify="center",
        ),
        dcc.Graph(id="cais-oeste-graph"),
        html.Br(),
    ]
)


@app.callback(
    dash.dependencies.Output("cais-oeste-content", "children"),
    [dash.dependencies.Input("cais-oeste-button", "value")],
)
@app.callback(
    dash.dependencies.Output("cais-oeste-graph", "figure"),
    [dash.dependencies.Input("cais-oeste-dropdown", "value"), dash.dependencies.Input("data-cais-oeste", "date")],
)
def plot_cais_oeste(value,data):
    fig = go.Figure()
    data = datetime.strptime(data, '%Y-%m-%d')
    # history, true_future, prediction = filter_dataframe(real, predictions, data)

    # index_history =history.index
    # index_target = true_future.index

    # if "Historico" in value:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=index_history,
    #             y=np.array(history.values)[:,0],
    #             name="Dados Passado",
    #             line=dict(color="blue", width=3),
    #         )
    #     )

    # if "Futuro" in value:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=index_target,
    #             y=np.array(true_future.values)[:,0],
    #             name="Dados Corretos",
    #             line=dict(color="lightblue", width=3),
    #         )
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=index_target,
    #             y=np.array(prediction.values)[:,0],
    #             name="Dados Estimados",
    #             line=dict(color="red", width=3),
    #         )
    #     )

    return fig.update_layout(
        title="Previsão para o dia " + data.strftime('%d/%m/%Y'),
        xaxis_title="Data",
        yaxis_title="Altura(m)",
        template=TEMPLATE,
    )

# Update the index
@app.callback(
    dash.dependencies.Output("page-content", "children"),
    [dash.dependencies.Input("url", "pathname")],
)
def display_page(pathname):
    if pathname.endswith("/galheta"):
        return galheta_layout
    elif pathname.endswith("/cais-oeste"):
        return cais_oeste_layout
    else:
        return index_page

if __name__ == "__main__":
    app.run_server(debug=False,port=8080)