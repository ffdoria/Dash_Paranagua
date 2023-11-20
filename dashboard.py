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
from PIL import Image




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
def filter_dataframe(df_real,df_predict, target_date_str,dias_historico=7,separar=True):
    pontos_tras = dias_historico*3*24
    pontos_frente = 24*3
    
    # Converta o índice para datetime se ainda não for
    if not isinstance(df_real.index, pd.DatetimeIndex):
        try:
            df_real.index = pd.to_datetime(df_real.index, format='%d/%m/%Y %H:%M')
        except ValueError:
            df_real.index = pd.to_datetime(df_real.index, format='%Y-%m-%d %H:%M:%S')
    if not isinstance(df_predict.index, pd.DatetimeIndex):
        try:
            df_predict.index = pd.to_datetime(df_predict.index, format='%d/%m/%Y %H:%M')
        except ValueError:
            df_predict.index = pd.to_datetime(df_predict.index, format='%Y-%m-%d %H:%M:%S')
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

large_text_style = {
    'fontSize': '1.5rem',  # Equivalent to large text, adjust as needed
    'lineHeight': '1.5',    # Adjust line height for readability
    'fontFamily': 'Georgia, serif'
}

path = ''
pil_image = Image.open(path+'loc_estacao.jpg')

#####Galheta 

galheta_predictions = pd.read_csv(path+'ALL_PREDICTIONS_GALHETA.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
galheta_predictions.rename(columns={'0': 'Altura'}, inplace=True)

galheta_real = pd.read_csv(path+'Erro_Galheta.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
galheta_real.rename(columns={'0': 'Altura'}, inplace=True)
galheta_real = fill_small_gaps_corrected(galheta_real, threshold=12)

galheta_ssh_real = pd.read_csv(path+'GALHETA_SSH.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
galheta_ssh_real.rename(columns={'0': 'Altura'}, inplace=True)
galheta_ssh_real = fill_small_gaps_corrected(galheta_ssh_real, threshold=12)

galheta_ssh_estimado = pd.read_csv(path+'SSH_PREDICT_GALHETA.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
galheta_ssh_estimado.rename(columns={'0': 'Altura'}, inplace=True)

galheta_teorico = pd.read_csv(path+'GALHETA_TEO.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
galheta_teorico.rename(columns={'0': 'Altura'}, inplace=True)

########CAis

cais_predictions = pd.read_csv(path+'ALL_PREDICTIONS_CAIS.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
cais_predictions.rename(columns={'0': 'Altura'}, inplace=True)

cais_real = pd.read_csv(path+'Erro_Cais_oeste.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
cais_real.rename(columns={'0': 'Altura'}, inplace=True)
cais_real = fill_small_gaps_corrected(cais_real, threshold=12)

cais_ssh_real = pd.read_csv(path+'CAIS_SSH.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
cais_ssh_real.rename(columns={'0': 'Altura'}, inplace=True)
cais_ssh_real = fill_small_gaps_corrected(cais_ssh_real, threshold=12)

cais_ssh_estimado = pd.read_csv(path+'SSH_PREDICT_CAIS.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
cais_ssh_estimado.rename(columns={'0': 'Altura'}, inplace=True)

cais_teorico = pd.read_csv(path+'CAIS_TEO.csv', decimal = '.', sep=',',thousands=" ",encoding='latin-1',index_col=0)
cais_teorico.rename(columns={'0': 'Altura'}, inplace=True)


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
        dbc.Row(
            dbc.Col(
                dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dcc.Link("Home", href=f"/{app_name}/", className="nav-link")),
                        dbc.NavItem(dcc.Link("Cais Oeste", href=f"/{app_name}/cais-oeste", className="nav-link")),
                        dbc.NavItem(dcc.Link("Galheta", href=f"/{app_name}/galheta", className="nav-link")),
                        # Add more links or dropdowns here if needed
                    ],
                    brand="Previsão do nível da maré no Porto de Paranaguá",
                    color="primary",  # Choose a color that matches your theme
                    dark=True,  # Set to False for a light-colored navbar
                    fluid=True,  # Set to True to make the navbar width align with the container
                    expand="lg"  # Adjust this for different screen sizes
                ),
                width=12  # This can be adjusted based on your layout needs
            ),
            justify="center"  # Adjust the alignment as needed
        ),
        html.Br(),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(width=1),
                dbc.Col(
                    html.P(
                        "Este site é um protótipo do algoritmo de previsão do nível da água do Porto de Paranaguá para auxiliar a atividade do local, usando técnicas de Machine Learning.",
                        style=large_text_style
                    ),
                    width=10
                ),
                dbc.Col(width=1),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(width=1),
                dbc.Col(
                    html.P(
                        "O algoritmo utiliza como base os dados passados, de 2015 a 2022, para poder realizar uma previsão que permite reduzir a margem de segurança de manobra dos navios na Baía de Paranaguá.",
                        style=large_text_style
                    ),
                    width=10
                ),
                dbc.Col(width=1),
            ],
            justify="left",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(width=1),
                dbc.Col(
                    html.P(
                        "As estações maregráficas que o algoritmo prevê são 1 (Cais Oeste) e 3 (Galheta).",
                        style=large_text_style
                    ),
                    width=10
                ),
                dbc.Col(width=1),
            ],
            justify="center",
        ),
        html.Br(),
        html.Div(
            html.Img(src=pil_image, style={'width': '70%', 'height': 'auto'}),
            style={'textAlign': 'center'}
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row(
            [   
                dbc.Col(width=1),
                dbc.Col(
                    html.Div(
                        children=[
                            "Para mais informações sobre o embasamento teórico utilizado para a criação desse dashboard, produto de um Trabalho de Conclusão de Curso da Escola Politécnica da USP, segue o link do arquivo da dissertação: ",
                            html.A("Aqui", href="http://bit.ly/TCC_POLI-USP_FD_LG", target="_blank")
                        ],
                        className="text-large"
                    )
                ),
                dbc.Col(width=1),           
            ],
            justify="left",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(width=1),
                dbc.Col(
                    html.Div(
                        children=[
                            "Para mais informações, entre em contato com os autores do projeto: ",
                            "Fernando Ferreira Doria - ",
                            html.A("ffdoria08@usp.br", href="mailto:ffdoria08@usp.br"),
                            ", Leonardo Guidetti Costa - ",
                            html.A("leonardoguidetti@usp.br", href="mailto:leonardoguidetti@usp.br"),
                            ", Eduardo Aoun Tannuri - ",
                            html.A("eduat@usp.br", href="mailto:eduat@usp.br"),'.'
                        ]
                    ),
                    width=10
                ),
                dbc.Col(width=1),
            ],
            justify="center",
        ),
    ]
)



"""GALHETA LAYOUT"""
galheta_layout = html.Div(
    [
        html.Div(id="galheta-content"),
        html.Br(),
        dbc.Row(
            dbc.Col(
                dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dcc.Link("Home", href=f"/{app_name}/", className="nav-link")),
                        dbc.NavItem(dcc.Link("Cais Oeste", href=f"/{app_name}/cais-oeste", className="nav-link")),
                        dbc.NavItem(dcc.Link("Galheta", href=f"/{app_name}/galheta", className="nav-link")),
                        # Add more links or dropdowns here if needed
                    ],
                    brand="Previsão do nível da maré no Porto de Paranaguá",
                    brand_href=f"/{app_name}/",
                    color="primary",  # Choose a color that matches your theme
                    dark=True,  # Set to False for a light-colored navbar
                    fluid=True,  # Set to True to make the navbar width align with the container
                    expand="lg"  # Adjust this for different screen sizes
                ),
                width=12  # This can be adjusted based on your layout needs
            ),
            justify="center"  # Adjust the alignment as needed
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
                        date=date(2021, 12, 27),
                        min_date_allowed=date(2021, 12, 27),
                        max_date_allowed=date(2022, 8, 22),
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
    history, true_future, prediction = filter_dataframe(galheta_real, galheta_predictions, data)

    index_history =history.index
    index_target = true_future.index

    if "Historico" in value:
        fig.add_trace(
            go.Scatter(
                x=index_history,
                y=np.array(history.values)[:,0],
                name="Histórico Maré Meteorológica",
                line=dict(color="blue", width=3),
            )
        )

    if "Futuro" in value:
        fig.add_trace(
            go.Scatter(
                x=index_target,
                y=np.array(true_future.values)[:,0],
                name="Maré Meteorológica Medida",
                line=dict(color="lightblue", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_target,
                y=np.array(prediction.values)[:,0],
                name="Maré Meteorológica Estimada",
                line=dict(color="red", width=3, dash="dash"),
            )
        )

    return fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Altura(m)",
        template=TEMPLATE,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.4,  # Adjust this value as needed to prevent overlap
            xanchor="center",
            x=0.5     # This centers the legend
        )
    )

@app.callback(
    dash.dependencies.Output('output-erro', 'children'),
    [dash.dependencies.Input("data-galheta", "date")]
)
def update_div(data):
    data = datetime.strptime(data, '%Y-%m-%d')
    history, true_future, prediction = filter_dataframe(galheta_real, galheta_predictions, data)

    try:
        MSE = round(mean_squared_error(true_future,prediction),3)
        RMSE =  round(np.sqrt(MSE),3)
    except ValueError:
        MSE = '-'
        RMSE = '-'

    return f'Erro dessa previsão é MSE: {MSE} e RMSE: {RMSE}.'


@app.callback(
    dash.dependencies.Output("galheta-graph-ssh", "figure"),
    [dash.dependencies.Input("galheta-dropdown-ssh", "value"), dash.dependencies.Input("data-galheta", "date")],
)
def plot_galheta_ssh(value,data):
    fig = go.Figure()
    data = datetime.strptime(data, '%Y-%m-%d')
    history_ssh, true_future_ssh, prediction_ssh = filter_dataframe(galheta_ssh_real, galheta_ssh_estimado, data)
    
    history_teo, _ , _ = filter_dataframe(galheta_teorico, galheta_ssh_estimado, data,separar=False)

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
                name="Histórico SSH Medido",
                line=dict(color="orange", width=3),
                showlegend=False
            )
        )
    

    if "Maré Estimada" in value:
        fig.add_trace(
            go.Scatter(
                x=index_target,
                y=np.array(prediction_ssh.values)[:,0],
                name="SSH Estimado",
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.4,  # Adjust this value as needed to prevent overlap
            xanchor="center",
            x=0.5     # This centers the legend
        )
    )
@app.callback(
    dash.dependencies.Output('output-erro-ssh', 'children'),
    [dash.dependencies.Input("data-galheta", "date")]
)
def update_div2(data):
    data = datetime.strptime(data, '%Y-%m-%d')
    history_ssh, true_future_ssh, prediction_ssh = filter_dataframe(galheta_ssh_real, galheta_ssh_estimado, data)
    _, predict_teo , _ = filter_dataframe(galheta_teorico, galheta_ssh_estimado, data,separar=False)

    try:
        MSE_teo = mean_squared_error(true_future_ssh,predict_teo)
        RMSE_teo =  np.sqrt(MSE_teo)
    except ValueError:
        MSE_teo = '-'
        RMSE_teo = '-'

    return f'Erro entre o SSH e a maré Astronômica é de MSE: {round(MSE_teo,3)} e RMSE: {round(RMSE_teo,3)}.'
    

"""CAIS OESTE LAYOUT"""
cais_oeste_layout = html.Div(
    [
        html.Div(id="cais-oeste-content"),
        html.Br(),
        dbc.Row(
            dbc.Col(
                dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dcc.Link("Home", href=f"/{app_name}/", className="nav-link")),
                        dbc.NavItem(dcc.Link("Cais Oeste", href=f"/{app_name}/cais-oeste", className="nav-link")),
                        dbc.NavItem(dcc.Link("Galheta", href=f"/{app_name}/galheta", className="nav-link")),
                        # Add more links or dropdowns here if needed
                    ],
                    brand="Previsão do nível da maré no Porto de Paranaguá",
                    brand_href=f"/{app_name}/",
                    color="primary",  # Choose a color that matches your theme
                    dark=True,  # Set to False for a light-colored navbar
                    fluid=True,  # Set to True to make the navbar width align with the container
                    expand="lg"  # Adjust this for different screen sizes
                ),
                width=12  # This can be adjusted based on your layout needs
            ),
            justify="center"  # Adjust the alignment as needed
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
                        id='data-cais',
                        date=date(2022, 1, 4),
                        min_date_allowed=date(2022, 1, 4),
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
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id='output-erro-cais'),
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
                        id="cais-dropdown-ssh",
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
        dcc.Graph(id="cais-graph-ssh"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id='output-erro-ssh-cais'),
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
    dash.dependencies.Output("cais-oeste-content", "children"),
    [dash.dependencies.Input("cais-oeste-button", "value")],
)
@app.callback(
    dash.dependencies.Output("cais-oeste-graph", "figure"),
    [dash.dependencies.Input("cais-oeste-dropdown", "value"), dash.dependencies.Input("data-cais", "date")],
)
def plot_cais_oeste(value,data_escolhida):
    fig = go.Figure()
    data = datetime.strptime(data_escolhida, '%Y-%m-%d')
    history, true_future, prediction = filter_dataframe(cais_real, cais_predictions, data)

    index_history =history.index
    index_target = true_future.index

    if "Historico" in value:
        fig.add_trace(
            go.Scatter(
                x=index_history,
                y=np.array(history.values)[:,0],
                name="Histórico Maré Meteorológica",
                line=dict(color="blue", width=3),
                showlegend=False
            )
        )

    if "Futuro" in value:
        fig.add_trace(
            go.Scatter(
                x=index_target,
                y=np.array(true_future.values)[:,0],
                name="Maré Meteorológica Medida",
                line=dict(color="lightblue", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_target,
                y=np.array(prediction.values)[:,0],
                name="Maré Meteorológica Estimada",
                line=dict(color="red", width=3, dash="dash"),
            )
        )

    return fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Altura(m)",
        template=TEMPLATE,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.4,  # Adjust this value as needed to prevent overlap
            xanchor="center",
            x=0.5     # This centers the legend
        )
    )

@app.callback(
    dash.dependencies.Output('output-erro-cais', 'children'),
    [dash.dependencies.Input("data-cais", "date")]
)
def update_div(data):
    data = datetime.strptime(data, '%Y-%m-%d')
    history, true_future, prediction = filter_dataframe(cais_real, cais_predictions, data)

    try:
        MSE = round(mean_squared_error(true_future,prediction),3)
        RMSE =  round(np.sqrt(MSE),3)
    except ValueError:
        MSE = '-'
        RMSE = '-'

    return f'Erro dessa previsão é MSE: {MSE} e RMSE: {RMSE}.'


@app.callback(
    dash.dependencies.Output("cais-graph-ssh", "figure"),
    [dash.dependencies.Input("cais-dropdown-ssh", "value"), dash.dependencies.Input("data-cais", "date")],
)
def plot_galheta_ssh(value,data):
    fig = go.Figure()
    data = datetime.strptime(data, '%Y-%m-%d')
    history_ssh, true_future_ssh, prediction_ssh = filter_dataframe(cais_ssh_real, cais_ssh_estimado, data)
    
    history_teo, _ , _ = filter_dataframe(cais_teorico, cais_ssh_estimado, data,separar=False)

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
                name="Histórico SSH Medido",
                line=dict(color="orange", width=3),
            )
        )
    

    if "Maré Estimada" in value:
        fig.add_trace(
            go.Scatter(
                x=index_target,
                y=np.array(prediction_ssh.values)[:,0],
                name="SSH Estimado",
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.4,  # Adjust this value as needed to prevent overlap
            xanchor="center",
            x=0.5     # This centers the legend
        )
    )
@app.callback(
    dash.dependencies.Output('output-erro-ssh-cais', 'children'),
    [dash.dependencies.Input("data-cais", "date")]
)
def update_div2(data):
    data = datetime.strptime(data, '%Y-%m-%d')
    history_ssh, true_future_ssh, prediction_ssh = filter_dataframe(cais_ssh_real, cais_ssh_estimado, data)
    _, predict_teo , _ = filter_dataframe(cais_teorico, cais_ssh_estimado, data,separar=False)

    try:
        MSE_teo = round(mean_squared_error(true_future_ssh,predict_teo),3)
        RMSE_teo =  round(np.sqrt(MSE_teo),3)
    except ValueError:
        MSE_teo = '-'
        RMSE_teo = '-'

    return f'Erro entre o SSH e a maré Astronômica é de MSE: {MSE_teo} e RMSE: {RMSE_teo}.'

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
    app.run_server(debug=False)