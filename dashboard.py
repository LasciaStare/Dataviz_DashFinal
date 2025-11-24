import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from consultas_postgres import cargar_datos_iniciales
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Paleta de colores del proyecto
COLORES = {
    'azul_profundo': '#003F5C',
    'azul_medio': '#2F4B7C',
    'purpura_frio': '#665191',
    'magenta_suave': '#A05195',
    'coral_rosado': '#D45087',
    'turquesa': '#45B7D1'
}

PALETTE = [COLORES['azul_profundo'], COLORES['azul_medio'], COLORES['purpura_frio'], 
           COLORES['magenta_suave'], COLORES['coral_rosado'], COLORES['turquesa']]

# Cargar datos UNA SOLA VEZ al inicio
print("Cargando datos desde PostgreSQL...")
try:
    datos = cargar_datos_iniciales()
    df_2007 = datos['df_2007']
    df_2022 = datos['df_2022']
    variables = datos['metadata']['variables']
    print("✓ Datos cargados exitosamente en memoria")
except Exception as e:
    print(f"✗ Error al cargar datos: {e}")
    df_2007 = pd.DataFrame()
    df_2022 = pd.DataFrame()
    variables = []

# Variables globales para almacenar los modelos entrenados
kmeans_2007 = None
kmeans_2022 = None
scaler_2007 = None
scaler_2022 = None
df_2007_clustered = None
df_2022_clustered = None
optimal_k = None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard del Proyecto Final "
server = app.server  


subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        html.H4('a. Definición del Problema a Resolver'),
        html.P('Tipo de problema: Agrupamiento (Clustering) no supervisado'),
        html.Ul([
            html.Li('Algoritmo: K-Means Clustering'),
            html.Li('Objetivo: Identificar arquetipos económicos globales comparando 2007 vs 2022'),
            html.Li('Variables de entrada: 14 indicadores macroeconómicos del Banco Mundial'),
            html.Li('Sin variable objetivo (aprendizaje no supervisado)')
        ])
    ]),
    dcc.Tab(label='b. Preparación de Datos', children=[
        html.H4('b. Preparación de los Datos'),
        html.P('Los datos fueron preprocesados siguiendo estos pasos:'),
        html.Ol([
            html.Li('Filtrado de países con más del 50% de datos faltantes'),
            html.Li('Imputación de valores faltantes usando la mediana'),
            html.Li('Detección y ajuste de outliers usando método IQR (factor 3x)'),
            html.Li('Normalización de variables usando StandardScaler (media=0, std=1)'),
            html.Li('Creación de datasets independientes para 2007 y 2022')
        ]),
        html.P('No se requiere división entrenamiento/prueba al ser clustering no supervisado. '\
               'La validación se realiza mediante métricas internas (Silhouette, Davies-Bouldin, Calinski-Harabasz).')
    ]),
    dcc.Tab(label='c. Selección del Modelo', children=[
        html.H4('c. Selección del Modelo: K-Means Clustering'),
        
        html.H5('Justificación', className='mt-3'),
        html.P('K-Means fue seleccionado por:'),
        html.Ul([
            html.Li('Eficiencia computacional con datasets grandes (200+ países)'),
            html.Li('Interpretabilidad: los centroides representan arquetipos económicos promedio'),
            html.Li('Adecuado para variables numéricas continuas'),
            html.Li('Permite comparación directa entre períodos usando el mismo k')
        ]),
        
        html.H5('Ecuación Matemática', className='mt-3'),
        html.P('K-Means minimiza la inercia (WCSS - Within-Cluster Sum of Squares):'),
        dcc.Markdown(r'''
        $$
        J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
        $$
        
        Donde:
        - $k$ = número de clusters
        - $C_i$ = cluster $i$
        - $\mu_i$ = centroide del cluster $i$
        - $x$ = punto de datos (país)
        ''', mathjax=True),
        
        html.H5('Método del Codo (Elbow Method)', className='mt-4 mb-3'),
        html.P('Selecciona el número óptimo de clusters para el año 2022:'),
        dcc.Graph(id='elbow-chart', style={'height': '450px'}),
        
        dbc.Row([
            dbc.Col([
                html.Label('Número de Clusters (k):', className='fw-bold mt-3'),
                dcc.Slider(
                    id='k-slider',
                    min=2,
                    max=8,
                    step=1,
                    value=4,
                    marks={i: str(i) for i in range(2, 9)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=8),
            dbc.Col([
                html.Button('Entrenar Modelos', id='train-button', 
                           className='btn btn-primary mt-4', n_clicks=0)
            ], width=4)
        ]),
        
        html.Div(id='training-status', className='mt-3')
    ]),
    
    dcc.Tab(label='d. Evaluación del Modelo', children=[
        html.H4('d. Entrenamiento y Evaluación del Modelo'),
        
        html.P('Los modelos K-Means se entrenan de forma independiente para cada año usando el mismo valor de k '\
               'para garantizar comparabilidad.'),
        
        html.H5('Proceso de Entrenamiento:', className='mt-3'),
        html.Ol([
            html.Li('Normalización de datos con StandardScaler'),
            html.Li('Inicialización: k-means++ (centroides optimizados)'),
            html.Li('Iteraciones: máximo 300'),
            html.Li('Semilla aleatoria: 42 (reproducibilidad)')
        ]),
        
        html.H5('Métricas de Evaluación', className='mt-4 mb-3'),
        html.Div(id='metrics-table'),
        
        html.H5('Distribución de Países por Cluster', className='mt-4 mb-3'),
        dbc.Row([
            dbc.Col([
                html.H6('Año 2007', className='text-center'),
                dcc.Graph(id='cluster-dist-2007', style={'height': '400px'})
            ], width=6),
            dbc.Col([
                html.H6('Año 2022', className='text-center'),
                dcc.Graph(id='cluster-dist-2022', style={'height': '400px'})
            ], width=6)
        ])
    ])
])


subtabs_resultados = dcc.Tabs([
    dcc.Tab(label='a. EDA', children=[
        html.H4('a. Análisis Exploratorio de Datos (EDA)', className='mb-4'),
        
        # Filtros
        dbc.Row([
            dbc.Col([
                html.Label('Selecciona el Año:', className='fw-bold'),
                dcc.Dropdown(
                    id='eda1-year-dropdown',
                    options=[
                        {'label': '2007 (Pre-Crisis Financiera)', 'value': 2007},
                        {'label': '2022 (Post-Pandemia)', 'value': 2022}
                    ],
                    value=2022,
                    clearable=False
                )
            ], width=4),
            dbc.Col([
                html.Label('Selecciona la Variable:', className='fw-bold'),
                dcc.Dropdown(
                    id='eda1-variable-dropdown',
                    options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in variables],
                    value=variables[0] if variables else None,
                    clearable=False
                )
            ], width=8)
        ], className='mb-4'),
        
        # Estadísticas descriptivas
        html.H5('Estadísticas Descriptivas', className='mt-4 mb-3'),
        html.Div(id='eda1-stats-table'),
        
        # Visualizaciones
        dbc.Row([
            dbc.Col([
                html.H5('Distribución de la Variable', className='mt-4 mb-3'),
                dcc.Graph(id='eda1-histogram', style={'height': '400px'})
            ], width=6),
            dbc.Col([
                html.H5('Box Plot', className='mt-4 mb-3'),
                dcc.Graph(id='eda1-boxplot', style={'height': '400px'})
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H5('Top 10 Países', className='mt-4 mb-3'),
                dcc.Graph(id='eda1-top10', style={'height': '450px'})
            ], width=12)
        ])
    ]),
    
    dcc.Tab(label='b. EDA 2', children=[
        html.H4('b. Análisis Exploratorio Comparativo', className='mb-4'),
        
        # Filtros
        dbc.Row([
            dbc.Col([
                html.Label('Variable Eje X:', className='fw-bold'),
                dcc.Dropdown(
                    id='eda2-var-x',
                    options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in variables],
                    value=variables[0] if variables else None,
                    clearable=False
                )
            ], width=6),
            dbc.Col([
                html.Label('Variable Eje Y:', className='fw-bold'),
                dcc.Dropdown(
                    id='eda2-var-y',
                    options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in variables],
                    value=variables[1] if len(variables) > 1 else None,
                    clearable=False
                )
            ], width=6)
        ], className='mb-4'),
        
        # Matriz de correlación
        dbc.Row([
            dbc.Col([
                html.H5('Matriz de Correlación', className='mb-3'),
                dcc.Dropdown(
                    id='eda2-corr-year',
                    options=[
                        {'label': '2007', 'value': 2007},
                        {'label': '2022', 'value': 2022}
                    ],
                    value=2022,
                    clearable=False,
                    style={'width': '200px'}
                ),
                dcc.Graph(id='eda2-correlation-matrix', style={'height': '700px'})
            ], width=12)
        ], className='mb-4'),
        
        # Scatter plots comparativos
        html.H5('Análisis de Dispersión Comparativo (2007 vs 2022)', className='mt-4 mb-3'),
        dbc.Row([
            dbc.Col([
                html.H6('Año 2007', className='text-center'),
                dcc.Graph(id='eda2-scatter-2007', style={'height': '450px'})
            ], width=6),
            dbc.Col([
                html.H6('Año 2022', className='text-center'),
                dcc.Graph(id='eda2-scatter-2022', style={'height': '450px'})
            ], width=6)
        ])
    ]),
    
    dcc.Tab(label='c. Visualización del Modelo', children=[
        html.H4('c. Visualización de Resultados del Modelo'),
        
        html.P('Visualización interactiva de los clusters identificados por el modelo K-Means.'),
        
        html.H5('Mapas Globales de Clusters', className='mt-4 mb-3'),
        dbc.Row([
            dbc.Col([
                html.H6('Año 2007', className='text-center'),
                dcc.Graph(id='choropleth-2007', style={'height': '500px'})
            ], width=6),
            dbc.Col([
                html.H6('Año 2022', className='text-center'),
                dcc.Graph(id='choropleth-2022', style={'height': '500px'})
            ], width=6)
        ]),
        
        html.H5('Análisis de Centroides', className='mt-4 mb-3'),
        html.P('Características económicas promedio de cada cluster:'),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='centroid-year-dropdown',
                    options=[
                        {'label': '2007', 'value': 2007},
                        {'label': '2022', 'value': 2022}
                    ],
                    value=2022,
                    clearable=False,
                    style={'width': '200px'}
                )
            ], width=3)
        ], className='mb-3'),
        dcc.Graph(id='centroid-heatmap', style={'height': '600px'}),
        
        html.H5('Scatter Plot 3D de Clusters', className='mt-4 mb-3'),
        dbc.Row([
            dbc.Col([
                html.Label('Variable X:', className='fw-bold'),
                dcc.Dropdown(id='scatter3d-x', 
                            options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in variables],
                            value=variables[0] if variables else None, clearable=False)
            ], width=4),
            dbc.Col([
                html.Label('Variable Y:', className='fw-bold'),
                dcc.Dropdown(id='scatter3d-y',
                            options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in variables],
                            value=variables[1] if len(variables) > 1 else None, clearable=False)
            ], width=4),
            dbc.Col([
                html.Label('Variable Z:', className='fw-bold'),
                dcc.Dropdown(id='scatter3d-z',
                            options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in variables],
                            value=variables[2] if len(variables) > 2 else None, clearable=False)
            ], width=4)
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([
                html.H6('Año 2007', className='text-center'),
                dcc.Graph(id='scatter3d-2007', style={'height': '500px'})
            ], width=6),
            dbc.Col([
                html.H6('Año 2022', className='text-center'),
                dcc.Graph(id='scatter3d-2022', style={'height': '500px'})
            ], width=6)
        ])
    ]),
    
    dcc.Tab(label='d. Indicadores del Modelo', children=[
        html.H4('d. Indicadores de Evaluación del Modelo'),
        
        html.P('Métricas de calidad del clustering para evaluar la cohesión y separación de los grupos.'),
        
        html.H5('Comparación de Métricas entre Años', className='mt-4 mb-3'),
        dcc.Graph(id='metrics-comparison', style={'height': '400px'}),
        
        html.H5('Tabla Detallada de Métricas', className='mt-4 mb-3'),
        html.Div(id='detailed-metrics-table'),
        
        html.H5('Interpretación de Métricas:', className='mt-4'),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Silhouette Score', className='fw-bold'),
                    dbc.CardBody([
                        html.P('Rango: [-1, 1]', className='mb-2'),
                        html.P('Interpretación:', className='fw-bold mb-1'),
                        html.Ul([
                            html.Li('Cercano a +1: Clusters bien definidos'),
                            html.Li('Cercano a 0: Clusters solapados'),
                            html.Li('Negativo: Asignaciones incorrectas')
                        ], style={'fontSize': '0.9rem'})
                    ])
                ], className='mb-3')
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Davies-Bouldin Index', className='fw-bold'),
                    dbc.CardBody([
                        html.P('Rango: [0, ∞)', className='mb-2'),
                        html.P('Interpretación:', className='fw-bold mb-1'),
                        html.Ul([
                            html.Li('Valores bajos: Mejor separación'),
                            html.Li('0: Clusters perfectamente separados'),
                            html.Li('Valores altos: Clusters solapados')
                        ], style={'fontSize': '0.9rem'})
                    ])
                ], className='mb-3')
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Calinski-Harabasz Score', className='fw-bold'),
                    dbc.CardBody([
                        html.P('Rango: [0, ∞)', className='mb-2'),
                        html.P('Interpretación:', className='fw-bold mb-1'),
                        html.Ul([
                            html.Li('Valores altos: Mejor clustering'),
                            html.Li('Mayor dispersión entre clusters'),
                            html.Li('Mayor compacidad intra-cluster')
                        ], style={'fontSize': '0.9rem'})
                    ])
                ], className='mb-3')
            ], width=4)
        ]),
        
        html.H5('Análisis de Inercia (WCSS)', className='mt-4 mb-3'),
        dcc.Graph(id='inertia-comparison', style={'height': '400px'})
    ]),
    dcc.Tab(label='e. Limitaciones', children=[
        html.H4('e. Limitaciones y Consideraciones Finales'),
        html.Ul([
            html.Li('Restricciones del análisis'),
            html.Li('Posibles mejoras futuras')
        ])
    ])
])


tabs = [
    dcc.Tab(label='1. Introducción', children=[
        html.H2('Introducción'),
        html.P('En un mundo cada vez más interconectado, comprender la dinámica de las economías nacionales y sus patrones '
               'de agrupación es fundamental para analistas, inversionistas y formuladores de políticas. La economía global '
               'ha atravesado por transformaciones profundas en las últimas dos décadas: desde la crisis financiera de 2008 '
               'que sacudió los cimientos del sistema financiero internacional, hasta la pandemia de COVID-19 que generó '
               'disrupciones sin precedentes en las cadenas de suministro y patrones de consumo.'),
        html.P('Este proyecto utiliza técnicas avanzadas de ciencia de datos y machine learning para identificar arquetipos '
               'económicos globales mediante clustering K-Means, comparando dos momentos clave: el año 2007, que representa '
               'la economía mundial antes de la Gran Recesión, y el año 2022, que refleja el panorama post-pandemia con sus '
               'nuevas dinámicas geopolíticas y económicas.'),
        html.P('A través del análisis de 14 indicadores macroeconómicos clave provenientes del Banco Mundial, este dashboard '
               'interactivo permite visualizar y explorar cómo han evolucionado los perfiles económicos de más de 200 países, '
               'identificando patrones ocultos, convergencias y divergencias en la estructura económica global. Los resultados '
               'ofrecen una perspectiva empírica y cuantitativa sobre las transformaciones que han redefinido el orden económico '
               'mundial en el siglo XXI.')
    ]),
    dcc.Tab(label='2. Contexto', children=[
        html.H2('Contexto'),
        html.P('Este proyecto analiza la estructura de la economía global mediante un enfoque de clustering no supervisado, '
               'comparando dos momentos históricos clave: el año 2007 (mundo pre-crisis financiera) y el año 2022 '
               '(mundo post-pandemia y de tensiones geopolíticas).'),
        html.H4('Fuente de los datos'),
        html.P('Los datos provienen del Banco Mundial (World Development Indicators - WDI), abarcando el período 2005-2024, '
               'con énfasis en los años 2007 y 2022 para el análisis comparativo.'),
        html.H4('Variables de interés'),
        html.Ul([
            html.Li('PIB per cápita (USD corrientes)'),
            html.Li('Crecimiento del PIB (% anual)'),
            html.Li('Inflación, deflactor del PIB (% anual)'),
            html.Li('Inversión extranjera directa, entradas netas (% del PIB)'),
            html.Li('Formación bruta de capital fijo (% del PIB)'),
            html.Li('Deuda externa total (USD corrientes)'),
            html.Li('Deuda externa (% del INB)'),
            html.Li('Servicio de la deuda (% de exportaciones)'),
            html.Li('Reservas totales en meses de importaciones'),
            html.Li('Saldo en cuenta corriente (% del PIB)'),
            html.Li('Desempleo total (% de la fuerza laboral)'),
            html.Li('Comercio (% del PIB)'),
            html.Li('Población total')
        ])
    ]),
    dcc.Tab(label='3. Planteamiento del Problema', children=[
        html.H2('Planteamiento del Problema'),
        html.P('La economía global ha experimentado transformaciones significativas en las últimas dos décadas, '
               'marcadas por crisis financieras, cambios en los patrones de comercio internacional, la pandemia de COVID-19, '
               'y tensiones geopolíticas. Estas transformaciones han reconfigurado las relaciones económicas entre países '
               'y han dado lugar a nuevos perfiles económicos.'),
        html.P('Es fundamental comprender cómo han evolucionado los arquetipos económicos de los países y si los grupos '
               'tradicionales (economías desarrolladas, mercados emergentes, países en desarrollo) se mantienen o han surgido '
               'nuevas configuraciones.'),
        html.H4('Pregunta Problema'),
        html.P(html.Strong('¿Cómo han cambiado los perfiles económicos de los países y su agrupación global, '
               'al comparar la estructura pre-crisis financiera (año 2007) con el panorama post-pandemia (año 2022), '
               'utilizando un modelo de clustering K-Means?'))
    ]),
    dcc.Tab(label='4. Objetivos y Justificación', children=[
        html.H2('Objetivos y Justificación'),
        html.H4('Objetivo General'),
        html.Ul([html.Li('Identificar y caracterizar los arquetipos económicos de los países a nivel global mediante clustering '
                         'K-Means, comparando los perfiles del año 2007 (pre-crisis financiera) con el año 2022 (post-pandemia), '
                         'para comprender las transformaciones en la estructura económica mundial.')]),
        html.H4('Objetivos Específicos'),
        html.Ul([
            html.Li('Aplicar el método del codo (Elbow Method) para determinar el número óptimo de clusters (k) que mejor '
                    'representa la estructura económica global en el año 2022.'),
            html.Li('Entrenar y comparar dos modelos K-Means independientes, uno para cada año de estudio (2007 y 2022), '
                    'utilizando el mismo valor de k para garantizar una comparación válida entre períodos.'),
            html.Li('Analizar los centroides de cada cluster para identificar las características económicas distintivas '
                    'de cada arquetipo y asignarles nombres descriptivos (ej: "Economías Desarrolladas", "Mercados Emergentes", etc.).'),
            html.Li('Visualizar los resultados mediante mapas del mundo interactivos (choropleth maps) que permitan identificar '
                    'la distribución geográfica de los clusters y las migraciones de países entre arquetipos en el período analizado.')
        ]),
        html.H4('Justificación'),
        html.P('La comprensión de los perfiles económicos y su evolución es crucial para la toma de decisiones en política '
               'económica internacional, inversiones, cooperación para el desarrollo y análisis de riesgos. Este proyecto '
               'proporciona una visión cuantitativa y visual de cómo la crisis financiera de 2008 y la pandemia de COVID-19 '
               'han reconfigurado el panorama económico global.'),
        html.P('El uso de técnicas de machine learning no supervisado (K-Means) permite descubrir patrones ocultos en los datos '
               'sin prejuicios previos, ofreciendo una clasificación empírica de los países basada exclusivamente en sus '
               'indicadores macroeconómicos. La comparación temporal entre 2007 y 2022 revela tendencias de convergencia o '
               'divergencia económica, proporcionando insights valiosos sobre la dinámica de la economía global en un período '
               'de profundas transformaciones.')
    ]),
    dcc.Tab(label='5. Marco Teórico', children=[
        html.H2('Marco Teórico'),
        html.P('Resumen de conceptos teóricos (definiciones formales) claves relacionados con el proyecto. Se pueden incluir referencias o citas.')
    ]),
    dcc.Tab(label='6. Metodología', children=[
        html.H2('Metodología'),
        subtabs_metodologia
    ]),
    dcc.Tab(label='7. Resultados y Análisis Final', children=[
        html.H2('Resultados y Análisis Final'),
        subtabs_resultados
    ]),
    dcc.Tab(label='8. Conclusiones', children=[
        html.H2('Conclusiones'),
        html.Ul([
            html.Li('Listar los principales hallazgos del proyecto'),
            html.Li('Relevancia de los resultados obtenidos'),
            html.Li('Aplicaciones futuras y recomendaciones')
        ])
    ])
]


app.layout = dbc.Container([
    html.H1("Dashboard del Proyecto Final ", className="text-center my-4"),
    dcc.Tabs(tabs)
], fluid=True)


# ==================== CALLBACKS PARA EDA 1 ====================

@app.callback(
    [Output('eda1-stats-table', 'children'),
     Output('eda1-histogram', 'figure'),
     Output('eda1-boxplot', 'figure'),
     Output('eda1-top10', 'figure')],
    [Input('eda1-year-dropdown', 'value'),
     Input('eda1-variable-dropdown', 'value')]
)
def update_eda1(year, variable):
    # Seleccionar el dataframe correcto
    df = df_2007 if year == 2007 else df_2022
    
    if df.empty or not variable:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No hay datos disponibles")
        return html.Div("No hay datos"), empty_fig, empty_fig, empty_fig
    
    # Estadísticas descriptivas
    stats = df[variable].describe()
    stats_table = dbc.Table([
        html.Thead(html.Tr([html.Th("Estadística"), html.Th("Valor")])),
        html.Tbody([
            html.Tr([html.Td("Media"), html.Td(f"{stats['mean']:.2f}")]),
            html.Tr([html.Td("Desviación Estándar"), html.Td(f"{stats['std']:.2f}")]),
            html.Tr([html.Td("Mínimo"), html.Td(f"{stats['min']:.2f}")]),
            html.Tr([html.Td("25%"), html.Td(f"{stats['25%']:.2f}")]),
            html.Tr([html.Td("Mediana (50%)"), html.Td(f"{stats['50%']:.2f}")]),
            html.Tr([html.Td("75%"), html.Td(f"{stats['75%']:.2f}")]),
            html.Tr([html.Td("Máximo"), html.Td(f"{stats['max']:.2f}")]),
            html.Tr([html.Td("Conteo"), html.Td(f"{int(stats['count'])}")])
        ])
    ], bordered=True, striped=True, hover=True, className='mt-2')
    
    # Histograma
    fig_hist = px.histogram(
        df, 
        x=variable,
        nbins=30,
        title=f'Distribución de {variable.replace("_", " ").title()} ({year})',
        labels={variable: variable.replace('_', ' ').title()},
        color_discrete_sequence=[COLORES['azul_profundo']]
    )
    fig_hist.update_layout(
        showlegend=False,
        template='plotly_white',
        font=dict(size=12),
        height=400
    )
    
    # Box plot
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=df[variable],
        name=variable.replace('_', ' ').title(),
        marker_color=COLORES['turquesa'],
        boxmean='sd'
    ))
    fig_box.update_layout(
        title=f'Box Plot - {variable.replace("_", " ").title()} ({year})',
        yaxis_title=variable.replace('_', ' ').title(),
        template='plotly_white',
        showlegend=False,
        height=400
    )
    
    # Top 10 países
    df_sorted = df.nlargest(10, variable)
    fig_top10 = px.bar(
        df_sorted,
        x=variable,
        y='Country Name',
        orientation='h',
        title=f'Top 10 Países - {variable.replace("_", " ").title()} ({year})',
        labels={variable: variable.replace('_', ' ').title(), 'Country Name': 'País'},
        color=variable,
        color_continuous_scale=[[0, COLORES['azul_profundo']], [1, COLORES['coral_rosado']]]
    )
    fig_top10.update_layout(
        template='plotly_white',
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        height=450
    )
    
    return stats_table, fig_hist, fig_box, fig_top10


# ==================== CALLBACKS PARA EDA 2 ====================

@app.callback(
    Output('eda2-correlation-matrix', 'figure'),
    Input('eda2-corr-year', 'value')
)
def update_correlation_matrix(year):
    df = df_2007 if year == 2007 else df_2022
    
    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No hay datos disponibles")
        return empty_fig
    
    # Calcular matriz de correlación solo con variables numéricas
    numeric_cols = [col for col in variables if col in df.columns]
    corr_matrix = df[numeric_cols].corr()
    
    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[col.replace('_', ' ').title() for col in corr_matrix.columns],
        y=[col.replace('_', ' ').title() for col in corr_matrix.columns],
        colorscale=[
            [0, COLORES['azul_profundo']],
            [0.5, '#FFFFFF'],
            [1, COLORES['coral_rosado']]
        ],
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlación")
    ))
    
    fig.update_layout(
        title=f'Matriz de Correlación - Año {year}',
        template='plotly_white',
        height=700,
        width=1000,
        xaxis={'tickangle': 45}
    )
    
    return fig


@app.callback(
    [Output('eda2-scatter-2007', 'figure'),
     Output('eda2-scatter-2022', 'figure')],
    [Input('eda2-var-x', 'value'),
     Input('eda2-var-y', 'value')]
)
def update_scatter_plots(var_x, var_y):
    if not var_x or not var_y:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Selecciona variables")
        return empty_fig, empty_fig
    
    # Scatter 2007
    fig_2007 = px.scatter(
        df_2007,
        x=var_x,
        y=var_y,
        hover_data=['Country Name'],
        title=f'{var_y.replace("_", " ").title()} vs {var_x.replace("_", " ").title()}',
        labels={
            var_x: var_x.replace('_', ' ').title(),
            var_y: var_y.replace('_', ' ').title()
        },
        trendline="ols",
        color_discrete_sequence=[COLORES['azul_medio']]
    )
    fig_2007.update_layout(template='plotly_white', showlegend=False, height=450)
    
    # Scatter 2022
    fig_2022 = px.scatter(
        df_2022,
        x=var_x,
        y=var_y,
        hover_data=['Country Name'],
        title=f'{var_y.replace("_", " ").title()} vs {var_x.replace("_", " ").title()}',
        labels={
            var_x: var_x.replace('_', ' ').title(),
            var_y: var_y.replace('_', ' ').title()
        },
        trendline="ols",
        color_discrete_sequence=[COLORES['magenta_suave']]
    )
    fig_2022.update_layout(template='plotly_white', showlegend=False, height=450)
    
    return fig_2007, fig_2022


# ==================== FUNCIONES AUXILIARES PARA CLUSTERING ====================

def train_kmeans_models(k):
    """Entrena modelos K-Means para ambos años"""
    global kmeans_2007, kmeans_2022, scaler_2007, scaler_2022
    global df_2007_clustered, df_2022_clustered, optimal_k
    
    optimal_k = k
    
    # Preparar datos 2007
    X_2007 = df_2007[variables].fillna(df_2007[variables].median())
    scaler_2007 = StandardScaler()
    X_2007_scaled = scaler_2007.fit_transform(X_2007)
    
    # Entrenar modelo 2007
    kmeans_2007 = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=42)
    clusters_2007 = kmeans_2007.fit_predict(X_2007_scaled)
    
    # Preparar datos 2022
    X_2022 = df_2022[variables].fillna(df_2022[variables].median())
    scaler_2022 = StandardScaler()
    X_2022_scaled = scaler_2022.fit_transform(X_2022)
    
    # Entrenar modelo 2022
    kmeans_2022 = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=42)
    clusters_2022 = kmeans_2022.fit_predict(X_2022_scaled)
    
    # Crear dataframes con clusters
    df_2007_clustered = df_2007.copy()
    df_2007_clustered['Cluster'] = clusters_2007
    
    df_2022_clustered = df_2022.copy()
    df_2022_clustered['Cluster'] = clusters_2022
    
    return True


def calculate_metrics(X_scaled, labels):
    """Calcula métricas de evaluación del clustering"""
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }


# ==================== CALLBACKS PARA METODOLOGÍA ====================

@app.callback(
    Output('elbow-chart', 'figure'),
    Input('k-slider', 'value')
)
def update_elbow_chart(selected_k):
    """Genera gráfico del método del codo"""
    if df_2022.empty:
        return go.Figure()
    
    X = df_2022[variables].fillna(df_2022[variables].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(K_range),
        y=inertias,
        mode='lines+markers',
        line=dict(color=COLORES['azul_medio'], width=3),
        marker=dict(size=10, color=COLORES['turquesa'])
    ))
    
    # Marcar el k seleccionado
    if selected_k in K_range:
        idx = list(K_range).index(selected_k)
        fig.add_trace(go.Scatter(
            x=[selected_k],
            y=[inertias[idx]],
            mode='markers',
            marker=dict(size=15, color=COLORES['coral_rosado'], 
                       line=dict(color='white', width=2)),
            name=f'k={selected_k}'
        ))
    
    fig.update_layout(
        title='Método del Codo - Inercia vs Número de Clusters',
        xaxis_title='Número de Clusters (k)',
        yaxis_title='Inercia (WCSS)',
        template='plotly_white',
        showlegend=True,
        height=450
    )
    
    return fig


@app.callback(
    Output('training-status', 'children'),
    Input('train-button', 'n_clicks'),
    Input('k-slider', 'value')
)
def train_models(n_clicks, k):
    """Entrena los modelos cuando se presiona el botón"""
    if n_clicks == 0:
        return html.Div([
            dbc.Alert('Selecciona el número de clusters y presiona "Entrenar Modelos"', color='info')
        ])
    
    try:
        train_kmeans_models(k)
        return html.Div([
            dbc.Alert([
                html.I(className='bi bi-check-circle me-2'),
                f'Modelos entrenados exitosamente con k={k} clusters'
            ], color='success')
        ])
    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.I(className='bi bi-exclamation-triangle me-2'),
                f'Error al entrenar modelos: {str(e)}'
            ], color='danger')
        ])


@app.callback(
    [Output('metrics-table', 'children'),
     Output('cluster-dist-2007', 'figure'),
     Output('cluster-dist-2022', 'figure')],
    Input('train-button', 'n_clicks')
)
def update_evaluation_tab(n_clicks):
    """Actualiza la pestaña de evaluación después del entrenamiento"""
    if n_clicks == 0 or kmeans_2007 is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return html.Div("Entrena los modelos primero"), empty_fig, empty_fig
    
    # Calcular métricas
    X_2007 = df_2007[variables].fillna(df_2007[variables].median())
    X_2007_scaled = scaler_2007.transform(X_2007)
    metrics_2007 = calculate_metrics(X_2007_scaled, df_2007_clustered['Cluster'])
    
    X_2022 = df_2022[variables].fillna(df_2022[variables].median())
    X_2022_scaled = scaler_2022.transform(X_2022)
    metrics_2022 = calculate_metrics(X_2022_scaled, df_2022_clustered['Cluster'])
    
    # Tabla de métricas
    metrics_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th('Métrica'),
            html.Th('2007'),
            html.Th('2022'),
            html.Th('Diferencia')
        ])),
        html.Tbody([
            html.Tr([
                html.Td('Silhouette Score'),
                html.Td(f"{metrics_2007['silhouette']:.4f}"),
                html.Td(f"{metrics_2022['silhouette']:.4f}"),
                html.Td(f"{metrics_2022['silhouette'] - metrics_2007['silhouette']:.4f}")
            ]),
            html.Tr([
                html.Td('Davies-Bouldin Index'),
                html.Td(f"{metrics_2007['davies_bouldin']:.4f}"),
                html.Td(f"{metrics_2022['davies_bouldin']:.4f}"),
                html.Td(f"{metrics_2022['davies_bouldin'] - metrics_2007['davies_bouldin']:.4f}")
            ]),
            html.Tr([
                html.Td('Calinski-Harabasz Score'),
                html.Td(f"{metrics_2007['calinski_harabasz']:.2f}"),
                html.Td(f"{metrics_2022['calinski_harabasz']:.2f}"),
                html.Td(f"{metrics_2022['calinski_harabasz'] - metrics_2007['calinski_harabasz']:.2f}")
            ]),
            html.Tr([
                html.Td('Inercia (WCSS)'),
                html.Td(f"{kmeans_2007.inertia_:.2f}"),
                html.Td(f"{kmeans_2022.inertia_:.2f}"),
                html.Td(f"{kmeans_2022.inertia_ - kmeans_2007.inertia_:.2f}")
            ])
        ])
    ], bordered=True, striped=True, hover=True)
    
    # Distribución de clusters 2007
    cluster_counts_2007 = df_2007_clustered['Cluster'].value_counts().sort_index()
    fig_2007 = go.Figure(data=[go.Bar(
        x=[f'Cluster {i}' for i in cluster_counts_2007.index],
        y=cluster_counts_2007.values,
        marker_color=PALETTE[:len(cluster_counts_2007)]
    )])
    fig_2007.update_layout(
        title='Distribución de Países por Cluster (2007)',
        xaxis_title='Cluster',
        yaxis_title='Número de Países',
        template='plotly_white',
        showlegend=False,
        height=400
    )
    
    # Distribución de clusters 2022
    cluster_counts_2022 = df_2022_clustered['Cluster'].value_counts().sort_index()
    fig_2022 = go.Figure(data=[go.Bar(
        x=[f'Cluster {i}' for i in cluster_counts_2022.index],
        y=cluster_counts_2022.values,
        marker_color=PALETTE[:len(cluster_counts_2022)]
    )])
    fig_2022.update_layout(
        title='Distribución de Países por Cluster (2022)',
        xaxis_title='Cluster',
        yaxis_title='Número de Países',
        template='plotly_white',
        showlegend=False,
        height=400
    )
    
    return metrics_table, fig_2007, fig_2022


# ==================== CALLBACKS PARA VISUALIZACIÓN DEL MODELO ====================

@app.callback(
    [Output('choropleth-2007', 'figure'),
     Output('choropleth-2022', 'figure')],
    Input('train-button', 'n_clicks')
)
def update_choropleth_maps(n_clicks):
    """Genera mapas de coropletas con los clusters"""
    if n_clicks == 0 or df_2007_clustered is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return empty_fig, empty_fig
    
    # Mapa 2007
    fig_2007 = px.choropleth(
        df_2007_clustered,
        locations='Country Code',
        color='Cluster',
        hover_name='Country Name',
        hover_data={var: ':.2f' for var in variables[:3]},
        color_continuous_scale=[[0, COLORES['azul_profundo']], 
                               [0.5, COLORES['turquesa']], 
                               [1, COLORES['coral_rosado']]],
        title='Distribución Global de Clusters - 2007'
    )
    fig_2007.update_layout(template='plotly_white', height=500)
    
    # Mapa 2022
    fig_2022 = px.choropleth(
        df_2022_clustered,
        locations='Country Code',
        color='Cluster',
        hover_name='Country Name',
        hover_data={var: ':.2f' for var in variables[:3]},
        color_continuous_scale=[[0, COLORES['azul_profundo']], 
                               [0.5, COLORES['turquesa']], 
                               [1, COLORES['coral_rosado']]],
        title='Distribución Global de Clusters - 2022'
    )
    fig_2022.update_layout(template='plotly_white', height=500)
    
    return fig_2007, fig_2022


@app.callback(
    Output('centroid-heatmap', 'figure'),
    [Input('train-button', 'n_clicks'),
     Input('centroid-year-dropdown', 'value')]
)
def update_centroid_heatmap(n_clicks, year):
    """Genera heatmap de centroides"""
    if n_clicks == 0 or kmeans_2007 is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return empty_fig
    
    kmeans = kmeans_2007 if year == 2007 else kmeans_2022
    scaler = scaler_2007 if year == 2007 else scaler_2022
    
    # Obtener centroides en escala original
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=centroids.T,
        x=[f'Cluster {i}' for i in range(optimal_k)],
        y=[var.replace('_', ' ').title() for var in variables],
        colorscale=[[0, COLORES['azul_profundo']], 
                   [0.5, '#FFFFFF'], 
                   [1, COLORES['coral_rosado']]],
        text=np.round(centroids.T, 2),
        texttemplate='%{text}',
        textfont={"size": 9},
        colorbar=dict(title="Valor")
    ))
    
    fig.update_layout(
        title=f'Características de los Centroides - Año {year}',
        xaxis_title='Cluster',
        yaxis_title='Variable Económica',
        template='plotly_white',
        height=600
    )
    
    return fig


@app.callback(
    [Output('scatter3d-2007', 'figure'),
     Output('scatter3d-2022', 'figure')],
    [Input('train-button', 'n_clicks'),
     Input('scatter3d-x', 'value'),
     Input('scatter3d-y', 'value'),
     Input('scatter3d-z', 'value')]
)
def update_scatter3d(n_clicks, var_x, var_y, var_z):
    """Genera scatter plots 3D con clusters"""
    if n_clicks == 0 or df_2007_clustered is None or not all([var_x, var_y, var_z]):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos y selecciona variables")
        return empty_fig, empty_fig
    
    # Scatter 3D 2007
    fig_2007 = px.scatter_3d(
        df_2007_clustered,
        x=var_x,
        y=var_y,
        z=var_z,
        color='Cluster',
        hover_name='Country Name',
        color_continuous_scale=[[0, COLORES['azul_profundo']], 
                               [0.5, COLORES['turquesa']], 
                               [1, COLORES['coral_rosado']]],
        title=f'Clusters en 3D - 2007',
        labels={
            var_x: var_x.replace('_', ' ').title(),
            var_y: var_y.replace('_', ' ').title(),
            var_z: var_z.replace('_', ' ').title()
        }
    )
    fig_2007.update_layout(template='plotly_white', height=500)
    
    # Scatter 3D 2022
    fig_2022 = px.scatter_3d(
        df_2022_clustered,
        x=var_x,
        y=var_y,
        z=var_z,
        color='Cluster',
        hover_name='Country Name',
        color_continuous_scale=[[0, COLORES['azul_profundo']], 
                               [0.5, COLORES['turquesa']], 
                               [1, COLORES['coral_rosado']]],
        title=f'Clusters en 3D - 2022',
        labels={
            var_x: var_x.replace('_', ' ').title(),
            var_y: var_y.replace('_', ' ').title(),
            var_z: var_z.replace('_', ' ').title()
        }
    )
    fig_2022.update_layout(template='plotly_white', height=500)
    
    return fig_2007, fig_2022


# ==================== CALLBACKS PARA INDICADORES DEL MODELO ====================

@app.callback(
    [Output('metrics-comparison', 'figure'),
     Output('detailed-metrics-table', 'children'),
     Output('inertia-comparison', 'figure')],
    Input('train-button', 'n_clicks')
)
def update_metrics_indicators(n_clicks):
    """Actualiza los indicadores y comparaciones de métricas"""
    if n_clicks == 0 or kmeans_2007 is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return empty_fig, html.Div("Entrena los modelos primero"), empty_fig
    
    # Calcular métricas
    X_2007 = df_2007[variables].fillna(df_2007[variables].median())
    X_2007_scaled = scaler_2007.transform(X_2007)
    metrics_2007 = calculate_metrics(X_2007_scaled, df_2007_clustered['Cluster'])
    
    X_2022 = df_2022[variables].fillna(df_2022[variables].median())
    X_2022_scaled = scaler_2022.transform(X_2022)
    metrics_2022 = calculate_metrics(X_2022_scaled, df_2022_clustered['Cluster'])
    
    # Gráfico de comparación de métricas
    fig_metrics = go.Figure()
    
    metrics_names = ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score']
    values_2007 = [metrics_2007['silhouette'], metrics_2007['davies_bouldin'], 
                   metrics_2007['calinski_harabasz'] / 100]  # Normalizar CH
    values_2022 = [metrics_2022['silhouette'], metrics_2022['davies_bouldin'], 
                   metrics_2022['calinski_harabasz'] / 100]
    
    fig_metrics.add_trace(go.Bar(
        name='2007',
        x=metrics_names,
        y=values_2007,
        marker_color=COLORES['azul_medio']
    ))
    fig_metrics.add_trace(go.Bar(
        name='2022',
        x=metrics_names,
        y=values_2022,
        marker_color=COLORES['coral_rosado']
    ))
    
    fig_metrics.update_layout(
        title='Comparación de Métricas entre Años',
        xaxis_title='Métrica',
        yaxis_title='Valor (normalizado)',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    # Tabla detallada
    detailed_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th('Métrica'),
            html.Th('2007'),
            html.Th('2022'),
            html.Th('Cambio Absoluto'),
            html.Th('Cambio Porcentual')
        ])),
        html.Tbody([
            html.Tr([
                html.Td('Silhouette Score'),
                html.Td(f"{metrics_2007['silhouette']:.4f}"),
                html.Td(f"{metrics_2022['silhouette']:.4f}"),
                html.Td(f"{metrics_2022['silhouette'] - metrics_2007['silhouette']:.4f}"),
                html.Td(f"{((metrics_2022['silhouette'] / metrics_2007['silhouette']) - 1) * 100:.2f}%")
            ]),
            html.Tr([
                html.Td('Davies-Bouldin Index'),
                html.Td(f"{metrics_2007['davies_bouldin']:.4f}"),
                html.Td(f"{metrics_2022['davies_bouldin']:.4f}"),
                html.Td(f"{metrics_2022['davies_bouldin'] - metrics_2007['davies_bouldin']:.4f}"),
                html.Td(f"{((metrics_2022['davies_bouldin'] / metrics_2007['davies_bouldin']) - 1) * 100:.2f}%")
            ]),
            html.Tr([
                html.Td('Calinski-Harabasz Score'),
                html.Td(f"{metrics_2007['calinski_harabasz']:.2f}"),
                html.Td(f"{metrics_2022['calinski_harabasz']:.2f}"),
                html.Td(f"{metrics_2022['calinski_harabasz'] - metrics_2007['calinski_harabasz']:.2f}"),
                html.Td(f"{((metrics_2022['calinski_harabasz'] / metrics_2007['calinski_harabasz']) - 1) * 100:.2f}%")
            ]),
            html.Tr([
                html.Td('Inercia (WCSS)'),
                html.Td(f"{kmeans_2007.inertia_:.2f}"),
                html.Td(f"{kmeans_2022.inertia_:.2f}"),
                html.Td(f"{kmeans_2022.inertia_ - kmeans_2007.inertia_:.2f}"),
                html.Td(f"{((kmeans_2022.inertia_ / kmeans_2007.inertia_) - 1) * 100:.2f}%")
            ])
        ])
    ], bordered=True, striped=True, hover=True)
    
    # Gráfico de inercia
    fig_inertia = go.Figure()
    fig_inertia.add_trace(go.Bar(
        x=['2007', '2022'],
        y=[kmeans_2007.inertia_, kmeans_2022.inertia_],
        marker_color=[COLORES['azul_medio'], COLORES['coral_rosado']],
        text=[f"{kmeans_2007.inertia_:.2f}", f"{kmeans_2022.inertia_:.2f}"],
        textposition='auto'
    ))
    fig_inertia.update_layout(
        title='Comparación de Inercia (WCSS) entre Años',
        xaxis_title='Año',
        yaxis_title='Inercia (menor es mejor)',
        template='plotly_white',
        showlegend=False,
        height=400
    )
    
    return fig_metrics, detailed_table, fig_inertia


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host="0.0.0.0", port=port)