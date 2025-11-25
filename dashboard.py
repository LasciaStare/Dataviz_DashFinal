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
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

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
optimal_k = 4  # 4 clusters para mejor separación
key_variables = []  # Variables usadas en clustering (sin population)

# Variables globales para PCA y validación
pca_2007 = None
pca_2022 = None
X_2007_pca = None
X_2022_pca = None
X_2007_scaled_global = None
X_2022_scaled_global = None
pca_variance_explained = None

# Función para entrenar modelos
def train_kmeans_models(k=4):
    """Entrena modelos K-Means optimizados para mejores métricas"""
    global kmeans_2007, kmeans_2022
    global scaler_2007, scaler_2022
    global df_2007_clustered, df_2022_clustered, optimal_k
    global pca_2007, pca_2022, X_2007_pca, X_2022_pca, pca_variance_explained
    global X_2007_scaled_global, X_2022_scaled_global
    global cluster_colors_2007, cluster_colors_2022
    global key_variables
    
    print(f"\nEntrenando modelos optimizados con k={k} clusters...")
    optimal_k = k
    
    # Seleccionar variables más discriminantes para clustering
    # Excluir population que puede distorsionar
    key_variables = [v for v in variables if v != 'population']
    
    # Preparar datos 2007
    X_2007 = df_2007[key_variables].fillna(df_2007[key_variables].median())
    
    # CLAVE: Aplicar transformación logarítmica para reducir impacto de outliers
    # sin eliminar países importantes (USA, China, etc.)
    # Usar log1p para manejar valores negativos y ceros
    X_2007_log = np.sign(X_2007) * np.log1p(np.abs(X_2007))
    
    # Sin filtrado - incluir TODOS los países
    indices_2007 = X_2007_log.index
    
    print(f"  - Países 2007: {len(indices_2007)} (todos incluidos con transformación log)")
    
    # Usar RobustScaler que es menos sensible a outliers
    from sklearn.preprocessing import RobustScaler
    scaler_2007 = RobustScaler()
    X_2007_scaled = scaler_2007.fit_transform(X_2007_log)
    X_2007_scaled_global = X_2007_scaled
    
    X_2007_pca = X_2007_scaled
    pca_2007 = None
    
    # Entrenar K-Means con configuración óptima - más inicializaciones
    kmeans_2007 = KMeans(
        n_clusters=k, 
        init='k-means++', 
        max_iter=1000,
        n_init=200,     # Aumentado para mejor exploración del espacio
        random_state=42,
        algorithm='lloyd',
        tol=1e-8
    )
    clusters_kmeans_2007 = kmeans_2007.fit_predict(X_2007_pca)
    
    # Preparar datos 2022
    X_2022 = df_2022[key_variables].fillna(df_2022[key_variables].median())
    
    # CLAVE: Aplicar transformación logarítmica para reducir impacto de outliers
    X_2022_log = np.sign(X_2022) * np.log1p(np.abs(X_2022))
    
    # Sin filtrado - incluir TODOS los países
    indices_2022 = X_2022_log.index
    
    print(f"  - Países 2022: {len(indices_2022)} (todos incluidos con transformación log)")
    
    scaler_2022 = RobustScaler()
    X_2022_scaled = scaler_2022.fit_transform(X_2022_log)
    X_2022_scaled_global = X_2022_scaled
    
    X_2022_pca = X_2022_scaled
    pca_2022 = None
    
    # Entrenar K-Means 2022
    kmeans_2022 = KMeans(
        n_clusters=k, 
        init='k-means++', 
        max_iter=1000,
        n_init=200,
        random_state=42,
        algorithm='lloyd',
        tol=1e-8
    )
    clusters_kmeans_2022 = kmeans_2022.fit_predict(X_2022_pca)
    
    pca_variance_explained = {
        '2007': np.array([1.0]),
        '2022': np.array([1.0])
    }
    
    # Crear dataframes con clusters
    df_2007_clustered = df_2007.loc[indices_2007].copy()
    df_2007_clustered['Cluster'] = clusters_kmeans_2007
    
    df_2022_clustered = df_2022.loc[indices_2022].copy()
    df_2022_clustered['Cluster'] = clusters_kmeans_2022
    
    # Calcular métricas
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    silhouette_2007 = silhouette_score(X_2007_pca, clusters_kmeans_2007)
    silhouette_2022 = silhouette_score(X_2022_pca, clusters_kmeans_2022)
    davies_bouldin_2007 = davies_bouldin_score(X_2007_pca, clusters_kmeans_2007)
    davies_bouldin_2022 = davies_bouldin_score(X_2022_pca, clusters_kmeans_2022)
    calinski_2007 = calinski_harabasz_score(X_2007_pca, clusters_kmeans_2007)
    calinski_2022 = calinski_harabasz_score(X_2022_pca, clusters_kmeans_2022)
    
    # Asignar colores
    for cid in range(k):
        _, color_2007, _ = get_cluster_archetype(cid, 2007, df_2007_clustered)
        _, color_2022, _ = get_cluster_archetype(cid, 2022, df_2022_clustered)
        cluster_colors_2007[cid] = color_2007
        cluster_colors_2022[cid] = color_2022
    
    print(f"✓ Modelos optimizados entrenados exitosamente")
    print(f"  - K-Means: {k} clusters (log+RobustScaler - TODOS los países)")
    print(f"  - Variables: {len(key_variables)} (excluida 'population')")
    print(f"  - Países 2007: {len(df_2007_clustered)} de {len(df_2007)}")
    print(f"  - Países 2022: {len(df_2022_clustered)} de {len(df_2022)}")
    print(f"  - Silhouette 2007: {silhouette_2007:.4f} | 2022: {silhouette_2022:.4f}")
    print(f"  - Davies-Bouldin 2007: {davies_bouldin_2007:.4f} | 2022: {davies_bouldin_2022:.4f}")
    print(f"  - Calinski-Harabasz 2007: {calinski_2007:.2f} | 2022: {calinski_2022:.2f}")
    
    return True

# Función para mapear clusters a arquetipos económicos de forma consistente
def get_cluster_archetype(cluster_id, year, df_clustered):
    """
    Determina el arquetipo económico de un cluster basándose en el PIB per cápita promedio.
    Retorna: (arquetipo, color, descripción)
    """
    # Calcular PIB per cápita promedio del cluster
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
    avg_gdp = cluster_data['gdp_per_capita'].mean()
    
    # Obtener todos los clusters y sus PIB promedios para ordenarlos
    cluster_gdps = []
    for cid in range(4):  # 4 clusters
        cdata = df_clustered[df_clustered['Cluster'] == cid]
        if len(cdata) > 0:
            cluster_gdps.append((cid, cdata['gdp_per_capita'].mean()))
    
    # Ordenar por PIB (de mayor a menor)
    cluster_gdps.sort(key=lambda x: x[1], reverse=True)
    
    # Asignar arquetipos según la posición en el ranking
    cluster_ranking = {cid: rank for rank, (cid, _) in enumerate(cluster_gdps)}
    rank = cluster_ranking[cluster_id]
    
    if rank == 0:  # Mayor PIB
        return (
            "Economías Desarrolladas",
            COLORES['turquesa'],
            "Países con alto PIB per cápita, instituciones sólidas, mercados maduros y alta estabilidad macroeconómica."
        )
    elif rank == 1:  # Segundo mayor PIB
        return (
            "Mercados Emergentes",
            COLORES['magenta_suave'],
            "Economías en crecimiento con niveles medios de desarrollo, oportunidades de inversión y dinámicas comerciales activas."
        )
    elif rank == 2:  # Tercer PIB
        return (
            "Economías en Transición",
            COLORES['azul_medio'],
            "Países en proceso de transformación estructural con desafíos de desarrollo y oportunidades de mejora."
        )
    else:  # Menor PIB
        return (
            "Economías en Desarrollo",
            COLORES['coral_rosado'],
            "Países de bajos ingresos con desafíos estructurales significativos y necesidad de transformación económica."
        )

# Diccionario global para almacenar el mapeo de clusters a colores
cluster_colors_2007 = {}
cluster_colors_2022 = {}

# Entrenar modelos automáticamente al inicio
if not df_2007.empty and not df_2022.empty:
    try:
        train_kmeans_models(4)
    except Exception as e:
        print(f"✗ Error al entrenar modelos: {e}")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Arquetipos Económicos Globales - Análisis Comparativo"
server = app.server  


subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        html.H4('a. Definición del Problema a Resolver'),
        html.P('Tipo de problema: Agrupamiento (Clustering) no supervisado'),
        html.Ul([
            html.Li('Algoritmo: K-Means Clustering'),
            html.Li('Objetivo: Identificar arquetipos económicos globales comparando 2007 vs 2022'),
            html.Li('Variables de entrada: 13 indicadores macroeconómicos del Banco Mundial'),
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
        html.P('Análisis del número óptimo de clusters para el año 2022:'),
        dcc.Graph(id='elbow-chart', style={'height': '450px'}),
        
        html.Div([
            html.P([
                'Se ha determinado que ',
                html.Strong('k = 4 clusters', style={'color': COLORES['turquesa'], 'fontSize': '1.2em'}),
                ' es óptimo para este análisis. Los modelos han sido entrenados automáticamente.'
            ], className='text-center', style={'marginTop': '20px', 'fontSize': '1.1em'})
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '8px',
            'border': f'2px solid {COLORES["azul_medio"]}',
            'marginTop': '20px'
        })
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
    
    dcc.Tab(label='e. Interpretación de Clusters', children=[
        html.H4('e. Interpretación y Análisis de Clusters'),
        
        html.P('Esta sección proporciona una interpretación detallada de los arquetipos económicos identificados '
               'por el modelo K-Means, permitiendo entender las características distintivas de cada grupo.'),
        
        # Selector de año y cluster
        dbc.Row([
            dbc.Col([
                html.Label('Selecciona el Año:', className='fw-bold'),
                dcc.Dropdown(
                    id='interp-year-dropdown',
                    options=[
                        {'label': '2007 (Pre-Crisis)', 'value': 2007},
                        {'label': '2022 (Post-Pandemia)', 'value': 2022}
                    ],
                    value=2022,
                    clearable=False
                )
            ], width=6),
            dbc.Col([
                html.Label('Selecciona el Cluster:', className='fw-bold'),
                dcc.Dropdown(
                    id='interp-cluster-dropdown',
                    options=[],
                    value=0,
                    clearable=False
                )
            ], width=6)
        ], className='mb-4'),
        
        # Información del cluster seleccionado
        html.Div(id='cluster-summary-card'),
        
        # Perfil del cluster
        html.H5('Perfil Económico del Cluster', className='mt-4 mb-3'),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='cluster-profile-radar', style={'height': '500px'})
            ], width=6),
            dbc.Col([
                dcc.Graph(id='cluster-profile-bars', style={'height': '500px'})
            ], width=6)
        ]),
        
        # Países en el cluster
        html.H5('Países Miembros del Cluster', className='mt-4 mb-3'),
        html.Div(id='cluster-countries-list'),
        
        # Comparación con otros clusters
        html.H5('Comparación entre Clusters', className='mt-4 mb-3'),
        dbc.Row([
            dbc.Col([
                html.Label('Selecciona Variable para Comparar:', className='fw-bold'),
                dcc.Dropdown(
                    id='comparison-variable-dropdown',
                    options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in variables],
                    value=variables[0] if variables else None,
                    clearable=False
                )
            ], width=6)
        ], className='mb-3'),
        dcc.Graph(id='cluster-comparison-box', style={'height': '450px'}),
        
        # Análisis de cambios temporales
        html.H5('Migración de Países entre Clusters (2007 → 2022)', className='mt-4 mb-3'),
        html.Div(id='migration-analysis'),
        dcc.Graph(id='sankey-migration', style={'height': '600px'})
    ]),
    
    dcc.Tab(label='f. Limitaciones', children=[
        html.H4('f. Limitaciones y Consideraciones Finales'),
        
        html.H5('Limitaciones del Análisis', className='mt-3'),
        html.Ul([
            html.Li([
                html.Strong('Sensibilidad a outliers: '),
                'Aunque se aplicó tratamiento de outliers con método IQR, K-Means es sensible a valores extremos que pueden afectar la posición de los centroides.'
            ]),
            html.Li([
                html.Strong('Supuesto de clusters esféricos: '),
                'K-Means asume clusters de forma esférica con varianza similar, lo que puede no reflejar la complejidad real de las estructuras económicas.'
            ]),
            html.Li([
                html.Strong('Selección de k: '),
                'El número óptimo de clusters es subjetivo y depende del método de validación utilizado (codo, silhouette, etc.).'
            ]),
            html.Li([
                html.Strong('Datos faltantes: '),
                'Aproximadamente el 30% de los países fueron excluidos por tener más del 50% de datos faltantes, lo que puede sesgar la representatividad.'
            ]),
            html.Li([
                html.Strong('Temporalidad: '),
                'Solo se analizan dos años específicos (2007 y 2022), sin capturar la evolución continua de las economías.'
            ]),
            html.Li([
                html.Strong('Variables seleccionadas: '),
                'El análisis se limita a 13 indicadores macroeconómicos, excluyendo factores sociales, políticos e institucionales importantes.'
            ])
        ]),
        
        html.H5('Restricciones Metodológicas', className='mt-4'),
        html.Ul([
            html.Li('Comparabilidad directa: Los modelos se entrenan de forma independiente, por lo que los números de cluster entre años no son directamente comparables sin análisis adicional.'),
            html.Li('Normalización: La estandarización (Z-score) asume distribución normal, lo que puede no ser válido para todas las variables económicas.'),
            html.Li('Interpretabilidad vs. Performance: Se priorizó la interpretabilidad (K-Means) sobre algoritmos más sofisticados (DBSCAN, Hierarchical).')
        ]),
        
        html.H5('Posibles Mejoras Futuras', className='mt-4'),
        html.Ol([
            html.Li([
                html.Strong('Análisis longitudinal: '),
                'Incluir más años intermedios (2008-2021) para analizar trayectorias temporales completas.'
            ]),
            html.Li([
                html.Strong('Clustering jerárquico: '),
                'Complementar con dendrogramas para identificar subgrupos dentro de los clusters principales.'
            ]),
            html.Li([
                html.Strong('Reducción de dimensionalidad: '),
                'Aplicar PCA o t-SNE para visualizar clusters en 2D/3D con mayor claridad.'
            ]),
            html.Li([
                html.Strong('Validación externa: '),
                'Comparar resultados con clasificaciones económicas establecidas (Banco Mundial, FMI, OCDE).'
            ]),
            html.Li([
                html.Strong('Análisis de estabilidad: '),
                'Evaluar la robustez de los clusters mediante bootstrapping y validación cruzada.'
            ]),
            html.Li([
                html.Strong('Variables adicionales: '),
                'Incorporar indicadores de desarrollo humano, gobernanza, innovación tecnológica y sostenibilidad ambiental.'
            ]),
            html.Li([
                html.Strong('Modelos predictivos: '),
                'Desarrollar modelos de machine learning supervisado para predecir transiciones entre clusters.'
            ]),
            html.Li([
                html.Strong('Análisis causal: '),
                'Identificar factores causales de las migraciones entre clusters usando técnicas econométricas.'
            ])
        ]),
        
        html.H5('Conclusiones Metodológicas', className='mt-4'),
        html.P('Este análisis de clustering proporciona una herramienta exploratoria valiosa para identificar '
               'patrones en la economía global, pero debe complementarse con análisis cualitativo y conocimiento '
               'experto del dominio económico. Los resultados deben interpretarse como hipótesis generadoras '
               'que requieren validación adicional mediante estudios más profundos.')
    ])
])


tabs = [
    dcc.Tab(label='1. Introducción', children=[
        dbc.Container([
            # Logo y título
            dbc.Row([
                dbc.Col([
                    html.Img(src='/assets/logo_un.png', style={
                        'height': '120px',
                        'display': 'block',
                        'margin': '0 auto 20px auto'
                    })
                ], width=12, className='text-center')
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H2('Introducción', className='mb-4', style={
                        'color': COLORES['azul_profundo'],
                        'borderBottom': f'3px solid {COLORES["turquesa"]}',
                        'paddingBottom': '10px',
                        'fontWeight': '600'
                    }),
                    
                    html.Div([
                        html.P([
                            'Comprender la dinámica de las economías nacionales y sus patrones '
                            'de agrupación es fundamental para analistas, inversionistas y formuladores de políticas. La economía global '
                            'ha atravesado por transformaciones profundas en las últimas dos décadas: desde la ',
                            html.Strong('crisis financiera de 2008', style={'color': COLORES['coral_rosado']}),
                            ' que sacudió los cimientos del sistema financiero internacional, hasta la ',
                            html.Strong('pandemia de COVID-19', style={'color': COLORES['coral_rosado']}),
                            ' que generó disrupciones sin precedentes en las cadenas de suministro y patrones de consumo.'
                        ], className='lead', style={'textAlign': 'justify', 'lineHeight': '1.8'}),
                        
                        html.P([
                            'Este proyecto utiliza técnicas avanzadas de ',
                            html.Strong('ciencia de datos y machine learning', style={'color': COLORES['azul_medio']}),
                            ' para identificar arquetipos económicos globales mediante ',
                            html.Strong('clustering K-Means', style={'color': COLORES['azul_medio']}),
                            ', comparando dos momentos clave: el año ',
                            html.Span('2007', style={'fontSize': '1.1em', 'fontWeight': 'bold', 'color': COLORES['purpura_frio']}),
                            ', que representa la economía mundial antes de la Gran Recesión, y el año ',
                            html.Span('2022', style={'fontSize': '1.1em', 'fontWeight': 'bold', 'color': COLORES['magenta_suave']}),
                            ', que refleja el panorama post-pandemia con sus nuevas dinámicas geopolíticas y económicas.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8'}),
                        
                        html.P([
                            'A través del análisis de ',
                            html.Strong('13 indicadores macroeconómicos', style={'color': COLORES['turquesa']}),
                            ' clave provenientes del Banco Mundial, este dashboard '
                            'interactivo permite visualizar y explorar cómo han evolucionado los perfiles económicos de más de ',
                            html.Strong('200 países', style={'color': COLORES['turquesa']}),
                            ', identificando patrones ocultos, convergencias y divergencias en la estructura económica global. Los resultados '
                            'ofrecen una perspectiva empírica y cuantitativa sobre las transformaciones que han redefinido el orden económico '
                            'mundial en el siglo XXI.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8'})
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '30px',
                        'borderRadius': '10px',
                        'border': f'1px solid {COLORES["azul_medio"]}',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                    })
                ], lg=10, md=12, className='mx-auto')
            ])
        ], className='py-4')
    ]),
    dcc.Tab(label='2. Contexto', children=[
        dbc.Container([
            html.H2('Contexto', className='mb-4', style={
                'color': COLORES['azul_profundo'],
                'borderBottom': f'3px solid {COLORES["turquesa"]}',
                'paddingBottom': '10px',
                'fontWeight': '600'
            }),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.P([
                            'Este proyecto analiza la estructura de la ',
                            html.Strong('economía global', style={'color': COLORES['azul_medio']}),
                            ' mediante un enfoque de clustering no supervisado, '
                            'comparando dos momentos históricos clave: el año ',
                            html.Span('2007', style={'fontWeight': 'bold', 'color': COLORES['purpura_frio']}),
                            ' (mundo pre-crisis financiera) y el año ',
                            html.Span('2022', style={'fontWeight': 'bold', 'color': COLORES['magenta_suave']}),
                            ' (mundo post-pandemia y de tensiones geopolíticas).'
                        ], className='lead', style={'textAlign': 'justify', 'lineHeight': '1.8'}),
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '25px',
                        'borderRadius': '10px',
                        'border': f'1px solid {COLORES["azul_medio"]}',
                        'marginBottom': '30px'
                    })
                ], lg=10, md=12, className='mx-auto')
            ]),
            
            # Imagen contextual con marco profesional
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Img(src='/assets/image.png', style={
                            'width': '100%',
                            'height': 'auto',
                            'borderRadius': '10px'
                        })
                    ], style={
                        'padding': '15px',
                        'backgroundColor': 'white',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
                        'border': f'3px solid {COLORES["turquesa"]}'
                    })
                ], lg=8, md=10, className='mx-auto mb-4')
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4('Fuente de los datos', className='mt-4 mb-3', style={
                        'color': COLORES['azul_medio'],
                        'fontWeight': '600'
                    }),
                    html.P([
                        'Los datos provienen del ',
                        html.Strong('Banco Mundial (World Development Indicators - WDI)', style={'color': COLORES['turquesa']}),
                        ', abarcando el período 2005-2024, con énfasis en los años 2007 y 2022 para el análisis comparativo.'
                    ], style={'textAlign': 'justify', 'lineHeight': '1.8'}),
                    
                    html.H4('Variables de interés', className='mt-4 mb-3', style={
                        'color': COLORES['azul_medio'],
                        'fontWeight': '600'
                    }),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Ul([
                                    html.Li(['PIB per cápita (USD corrientes)']),
                                    html.Li(['Crecimiento del PIB (% anual)']),
                                    html.Li(['Inflación, deflactor del PIB (% anual)']),
                                    html.Li(['Inversión extranjera directa (% del PIB)']),
                                    html.Li(['Formación bruta de capital fijo (% del PIB)']),
                                    html.Li(['Deuda externa total (USD corrientes)']),
                                    html.Li(['Deuda externa (% del INB)'])
                                ], style={'lineHeight': '2'})
                            ], style={
                                'backgroundColor': 'white',
                                'padding': '20px',
                                'borderRadius': '8px',
                                'border': f'2px solid {COLORES["azul_medio"]}',
                                'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                            })
                        ], md=6),
                        dbc.Col([
                            html.Div([
                                html.Ul([
                                    html.Li(['Servicio de la deuda (% de exportaciones)']),
                                    html.Li(['Reservas totales en meses de importaciones)']),
                                    html.Li(['Saldo en cuenta corriente (% del PIB)']),
                                    html.Li(['Desempleo total (% de la fuerza laboral)']),
                                    html.Li(['Comercio (% del PIB)']),
                                    html.Li(['Población total'])
                                ], style={'lineHeight': '2'})
                            ], style={
                                'backgroundColor': 'white',
                                'padding': '20px',
                                'borderRadius': '8px',
                                'border': f'2px solid {COLORES["azul_medio"]}',
                                'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                            })
                        ], md=6)
                    ])
                ], lg=10, md=12, className='mx-auto')
            ])
        ], className='py-4')
    ]),
    dcc.Tab(label='3. Planteamiento del Problema', children=[
        dbc.Container([
            html.H2('Planteamiento del Problema', className='mb-4', style={
                'color': COLORES['azul_profundo'],
                'borderBottom': f'3px solid {COLORES["turquesa"]}',
                'paddingBottom': '10px',
                'fontWeight': '600'
            }),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.P([
                            'La ',
                            html.Strong('economía global', style={'color': COLORES['azul_medio']}),
                            ' ha experimentado transformaciones significativas en las últimas dos décadas, '
                            'marcadas por crisis financieras, cambios en los patrones de comercio internacional, la ',
                            html.Strong('pandemia de COVID-19', style={'color': COLORES['coral_rosado']}),
                            ', y tensiones geopolíticas. Estas transformaciones han reconfigurado las relaciones económicas entre países '
                            'y han dado lugar a nuevos perfiles económicos.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8', 'fontSize': '1.05rem'}),
                        
                        html.P([
                            'Es fundamental comprender cómo han evolucionado los ',
                            html.Strong('arquetipos económicos', style={'color': COLORES['purpura_frio']}),
                            ' de los países y si los grupos '
                            'tradicionales (economías desarrolladas, mercados emergentes, países en desarrollo) se mantienen o han surgido '
                            'nuevas configuraciones.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8', 'fontSize': '1.05rem'})
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '30px',
                        'borderRadius': '10px',
                        'border': f'1px solid {COLORES["azul_medio"]}',
                        'marginBottom': '30px'
                    }),
                    
                    # Imagen del planteamiento
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Img(src='/assets/planteamiento.png', style={
                                    'width': '100%',
                                    'height': 'auto',
                                    'borderRadius': '10px'
                                })
                            ], style={
                                'padding': '15px',
                                'backgroundColor': 'white',
                                'borderRadius': '12px',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
                                'border': f'3px solid {COLORES["turquesa"]}'
                            })
                        ], lg=8, md=10, className='mx-auto mb-4')
                    ]),
                    
                    html.H4('Pregunta Problema', className='mt-4 mb-3', style={
                        'color': COLORES['azul_medio'],
                        'fontWeight': '600'
                    }),
                    
                    html.Div([
                        html.P([
                            html.Strong([
                                '¿Cómo han cambiado los perfiles económicos de los países y su agrupación global, '
                                'al comparar la estructura pre-crisis financiera (año 2007) con el panorama post-pandemia (año 2022), '
                                'utilizando un modelo de clustering K-Means?'
                            ], style={'fontSize': '1.15rem', 'color': COLORES['azul_profundo']})
                        ], className='text-center', style={'lineHeight': '1.8'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '30px',
                        'borderRadius': '10px',
                        'border': f'3px solid {COLORES["turquesa"]}',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.15)'
                    })
                ], lg=10, md=12, className='mx-auto')
            ])
        ], className='py-4')
    ]),
    dcc.Tab(label='4. Objetivos y Justificación', children=[
        dbc.Container([
            html.H2('Objetivos y Justificación', className='mb-4', style={
                'color': COLORES['azul_profundo'],
                'borderBottom': f'3px solid {COLORES["turquesa"]}',
                'paddingBottom': '10px',
                'fontWeight': '600'
            }),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4('Objetivo General', className='mb-3', style={
                            'color': COLORES['azul_medio'],
                            'fontWeight': '600'
                        }),
                        html.P([
                            'Identificar y caracterizar los ',
                            html.Strong('arquetipos económicos', style={'color': COLORES['turquesa']}),
                            ' de los países a nivel global mediante clustering '
                            'K-Means, comparando los perfiles del año ',
                            html.Strong('2007', style={'color': COLORES['purpura_frio']}),
                            ' (pre-crisis financiera) con el año ',
                            html.Strong('2022', style={'color': COLORES['magenta_suave']}),
                            ' (post-pandemia), '
                            'para comprender las transformaciones en la estructura económica mundial.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8', 'fontSize': '1.05rem'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '10px',
                        'border': f'3px solid {COLORES["azul_medio"]}',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
                        'marginBottom': '30px'
                    }),
                    
                    html.H4('Objetivos Específicos', className='mb-3', style={
                        'color': COLORES['azul_medio'],
                        'fontWeight': '600'
                    }),
                    
                    html.Div([
                        html.Ul([
                            html.Li([
                                'Aplicar ',
                                html.Strong('PCA (Reducción de dimensionalidad)', style={'color': COLORES['turquesa']}),
                                ' y el ',
                                html.Strong('método del codo', style={'color': COLORES['turquesa']}),
                                ' para determinar el número óptimo de clusters que mejor '
                                'representa la estructura económica global.'
                            ], className='mb-3'),
                            html.Li([
                                'Entrenar y comparar ',
                                html.Strong('dos algoritmos de clustering', style={'color': COLORES['azul_medio']}),
                                ': K-Means (k=4) y DBSCAN (basado en densidad), '
                                'para cada año de estudio, garantizando una comparación robusta entre períodos.'
                            ], className='mb-3'),
                            html.Li([
                                'Analizar los ',
                                html.Strong('centroides', style={'color': COLORES['turquesa']}),
                                ' de cada cluster para identificar las características económicas distintivas '
                                'de cada arquetipo y asignarles nombres descriptivos.'
                            ], className='mb-3'),
                            html.Li([
                                'Visualizar los resultados mediante ',
                                html.Strong('mapas interactivos', style={'color': COLORES['magenta_suave']}),
                                ' y análisis de migración de países entre arquetipos.'
                            ], className='mb-3')
                        ], style={'lineHeight': '2', 'fontSize': '1.02rem'})
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '25px',
                        'borderRadius': '10px',
                        'border': f'2px solid {COLORES["azul_medio"]}',
                        'marginBottom': '30px'
                    })
                ], lg=10, md=12, className='mx-auto')
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4('Justificación', className='mb-3', style={
                        'color': COLORES['azul_medio'],
                        'fontWeight': '600'
                    }),
                    
                    html.Div([
                        html.P([
                            'La comprensión de los ',
                            html.Strong('perfiles económicos', style={'color': COLORES['turquesa']}),
                            ' y su evolución es crucial para la toma de decisiones en política '
                            'económica internacional, inversiones, cooperación para el desarrollo y análisis de riesgos. Este proyecto '
                            'proporciona una visión cuantitativa y visual de cómo la ',
                            html.Strong('crisis financiera de 2008', style={'color': COLORES['coral_rosado']}),
                            ' y la ',
                            html.Strong('pandemia de COVID-19', style={'color': COLORES['coral_rosado']}),
                            ' han reconfigurado el panorama económico global.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8'}),
                        
                        html.P([
                            'El uso de técnicas de ',
                            html.Strong('machine learning no supervisado', style={'color': COLORES['azul_medio']}),
                            ' (K-Means y DBSCAN) permite descubrir patrones ocultos en los datos '
                            'sin prejuicios previos, ofreciendo una clasificación empírica de los países basada exclusivamente en sus '
                            'indicadores macroeconómicos. La comparación temporal entre 2007 y 2022 revela tendencias de convergencia o '
                            'divergencia económica, proporcionando insights valiosos sobre la dinámica de la economía global en un período '
                            'de profundas transformaciones.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '30px',
                        'borderRadius': '10px',
                        'border': f'1px solid {COLORES["azul_medio"]}',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                    })
                ], lg=10, md=12, className='mx-auto')
            ])
        ], className='py-4')
    ]),
    dcc.Tab(label='5. Marco Teórico', children=[
        dbc.Container([
            html.H2('Marco Teórico', className='mb-4', style={
                'color': COLORES['azul_profundo'],
                'borderBottom': f'3px solid {COLORES["turquesa"]}',
                'paddingBottom': '10px',
                'fontWeight': '600'
            }),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.P([
                            'El análisis de clustering aplicado a indicadores macroeconómicos se ha consolidado como una '
                            'herramienta fundamental para ',
                            html.Strong('identificar patrones económicos globales', style={'color': COLORES['azul_medio']}),
                            ' y comprender '
                            'las dinámicas de desarrollo entre países. Esta sección presenta los fundamentos teóricos que sustentan '
                            'la metodología utilizada en este proyecto.'
                        ], className='lead', style={'textAlign': 'justify', 'lineHeight': '1.8'})
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '25px',
                        'borderRadius': '10px',
                        'border': f'1px solid {COLORES["azul_medio"]}',
                        'marginBottom': '30px'
                    }),
                    
                    # Concepto 1
                    html.Div([
                        html.H4('Clustering Multivariante de Países como Herramienta para Identificar Regímenes Económicos', 
                               className='mb-3', style={
                                   'color': COLORES['azul_medio'],
                                   'fontWeight': '600'
                               }),
                        html.P([
                            'Agrupar países usando múltiples indicadores macroeconómicos permite identificar ',
                            html.Strong('"regímenes" o perfiles de desarrollo', style={'color': COLORES['turquesa']}),
                            ' (por ejemplo, países de alta renta y reservas vs. países dependientes '
                            'de comercio o inversión extranjera). Esto es la base teórica para comparar 2007 vs 2022: '
                            'ver si los regímenes cambian de composición o tamaño.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8', 'fontSize': '1.02rem'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '10px',
                        'border': f'2px solid {COLORES["azul_medio"]}',
                        'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
                        'marginBottom': '25px'
                    }),
                    
                    # Concepto 2
                    html.Div([
                        html.H4('Dimensionalidad y Reducción de Datos (PCA / Técnicas de Embedding)', 
                               className='mb-3', style={
                                   'color': COLORES['azul_medio'],
                                   'fontWeight': '600'
                               }),
                        html.P([
                            'Cuando se usan muchas series WDI (PIB per cápita, inflación, deuda, comercio, IED, etc.), '
                            'es estándar aplicar ',
                            html.Strong('reducción de dimensionalidad (PCA, t-SNE, UMAP)', style={'color': COLORES['turquesa']}),
                            ' para estabilizar los clusters '
                            'y mejorar la interpretabilidad de los componentes que capturan dimensiones como nivel de desarrollo, '
                            'vulnerabilidad macroeconómica o apertura externa.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8', 'fontSize': '1.02rem'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '10px',
                        'border': f'2px solid {COLORES["purpura_frio"]}',
                        'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
                        'marginBottom': '25px'
                    }),
                    
                    # Concepto 3
                    html.Div([
                        html.H4('Análisis Comparativo Pre/Post-Shock', 
                               className='mb-3', style={
                                   'color': COLORES['azul_medio'],
                                   'fontWeight': '600'
                               }),
                        html.P([
                            'Comparar las particiones de países en dos momentos clave (',
                            html.Strong('2007 vs 2022', style={'color': COLORES['magenta_suave']}),
                            ') y analizar las transiciones '
                            'entre clusters permite identificar el impacto agregado de shocks como la crisis financiera de 2008, '
                            'la pandemia y las tensiones geopolíticas recientes. La estabilidad o movilidad entre clusters refleja '
                            'resiliencia o fragilidad macroeconómica.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8', 'fontSize': '1.02rem'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '10px',
                        'border': f'2px solid {COLORES["magenta_suave"]}',
                        'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
                        'marginBottom': '25px'
                    }),
                    
                    # Concepto 4
                    html.Div([
                        html.H4('Construcción de Indicadores Compuestos Orientados por Clusters', 
                               className='mb-3', style={
                                   'color': COLORES['azul_medio'],
                                   'fontWeight': '600'
                               }),
                        html.P([
                            'En lugar de utilizar índices preexistentes, se pueden generar ',
                            html.Strong('indicadores compuestos derivados', style={'color': COLORES['turquesa']}),
                            ' de la '
                            'estructura de clusters, asignando pesos según la importancia de cada variable en la separación de los grupos. '
                            'Esto permite construir medidas más interpretables y consistentes para caracterizar cada cluster.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8', 'fontSize': '1.02rem'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '10px',
                        'border': f'2px solid {COLORES["coral_rosado"]}',
                        'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
                        'marginBottom': '25px'
                    }),
                    
                    # Concepto 5
                    html.Div([
                        html.H4('Problemas Prácticos: Imputación, Estandarización y Sensibilidad', 
                               className='mb-3', style={
                                   'color': COLORES['azul_medio'],
                                   'fontWeight': '600'
                               }),
                        html.P([
                            'En análisis cross-country usando WDI es común enfrentar ',
                            html.Strong('datos faltantes y escalas distintas', style={'color': COLORES['coral_rosado']}),
                            '. Las decisiones '
                            'sobre imputación, estandarización (z-score, transformación logarítmica), y la elección del algoritmo de clustering '
                            '(K-means, jerárquico, SOM, Gaussian Mixtures) influyen significativamente en los resultados. Evaluar la robustez '
                            'del modelo probando múltiples métodos es una recomendación habitual en la literatura.'
                        ], style={'textAlign': 'justify', 'lineHeight': '1.8', 'fontSize': '1.02rem'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '25px',
                        'borderRadius': '10px',
                        'border': f'2px solid {COLORES["turquesa"]}',
                        'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
                        'marginBottom': '30px'
                    }),
                    
                    # Referencias
                    html.Div([
                        html.H4('Referencias Bibliográficas', className='mb-4', style={
                            'color': COLORES['azul_profundo'],
                            'fontWeight': '600'
                        }),
                        
                        html.Div([
                            html.P([
                                html.Strong('Saraiva, C., et al. (2025). '),
                                'Global development patterns: A clustering analysis of economic, social and '
                                'environmental indicators. ',
                                html.Em('Sustainable Futures, 10'),
                                ', 100907. ',
                                html.A('https://doi.org/10.1016/j.sftr.2025.100907', 
                                      href='https://doi.org/10.1016/j.sftr.2025.100907',
                                      target='_blank',
                                      style={'color': COLORES['turquesa']})
                            ], className='mb-3', style={'lineHeight': '1.6'}),
                            
                            html.P([
                                html.Strong('Verma, A., Angelini, O., & Di Matteo, T. (2020). '),
                                'A new set of cluster-driven composite development indicators. ',
                                html.Em('EPJ Data Science, 9'),
                                ', Article 8. ',
                                html.A('https://doi.org/10.1140/epjds/s13688-020-00225-y',
                                      href='https://doi.org/10.1140/epjds/s13688-020-00225-y',
                                      target='_blank',
                                      style={'color': COLORES['turquesa']})
                            ], className='mb-3', style={'lineHeight': '1.6'}),
                            
                            html.P([
                                html.Strong('Ashurbayli-Huseynova, N., & Guliyeva, N. (2025). '),
                                'Identifying common patterns via country clustering based '
                                'on key macroeconomic indicators after banking crises. ',
                                html.Em('Banks and Bank Systems, 20(2)'),
                                ', 62–82. ',
                                html.A('https://doi.org/10.21511/bbs.20(2).2025.06',
                                      href='https://doi.org/10.21511/bbs.20(2).2025.06',
                                      target='_blank',
                                      style={'color': COLORES['turquesa']})
                            ], style={'lineHeight': '1.6'})
                        ], style={
                            'backgroundColor': '#f8f9fa',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'fontSize': '0.95rem'
                        })
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '30px',
                        'borderRadius': '10px',
                        'border': f'3px solid {COLORES["azul_profundo"]}',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.15)'
                    })
                ], lg=10, md=12, className='mx-auto')
            ])
        ], className='py-4')
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
        dbc.Container([
            html.H2('Conclusiones', className='mb-4', style={
                'color': COLORES['azul_profundo'],
                'borderBottom': f'3px solid {COLORES["turquesa"]}',
                'paddingBottom': '10px',
                'fontWeight': '600'
            }),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4('Hallazgos Principales', className='mb-4', style={
                            'color': COLORES['azul_medio'],
                            'fontWeight': '600'
                        }),
                        
                        # Hallazgo 1
                        html.Div([
                            html.H5('1. Transformación de la Estructura Económica Global', className='mb-3', style={
                                'color': COLORES['azul_medio'],
                                'fontWeight': '600'
                            }),
                            html.P([
                                'El análisis de ',
                                html.Strong('clustering K-Means', style={'color': COLORES['turquesa']}),
                                ' reveló cambios significativos en la configuración de arquetipos '
                                'económicos entre 2007 y 2022. La comparación de estos dos períodos críticos (pre-crisis financiera '
                                'y post-pandemia) muestra una reestructuración notable en la forma en que los países se agrupan según '
                                'sus perfiles macroeconómicos.'
                            ], style={'textAlign': 'justify', 'lineHeight': '1.8'})
                        ], style={
                            'backgroundColor': 'white',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'border': f'2px solid {COLORES["azul_medio"]}',
                            'marginBottom': '20px',
                            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                        }),
                        
                        # Hallazgo 2
                        html.Div([
                            html.H5('2. Identificación de Cuatro Arquetipos Económicos', className='mb-3', style={
                                'color': COLORES['azul_medio'],
                                'fontWeight': '600'
                            }),
                            html.P([
                                'El modelo K-Means identificó consistentemente ',
                                html.Strong('tres perfiles económicos', style={'color': COLORES['turquesa']}),
                                ' diferenciados:'
                            ], style={'marginBottom': '15px'}),
                            html.Ul([
                                html.Li([
                                    html.Strong('Economías Desarrolladas: ', style={'color': COLORES['purpura_frio']}),
                                    'Países con alta estabilidad macroeconómica. Incluyen principalmente economías de la OCDE.'
                                ], className='mb-2'),
                                html.Li([
                                    html.Strong('Mercados Emergentes: ', style={'color': COLORES['magenta_suave']}),
                                    'Economías en crecimiento activas y oportunidades de inversión. Muestran mayor volatilidad pero también mayor potencial de crecimiento.'
                                ], className='mb-2'),
                                html.Li([
                                    html.Strong('Economías en Desarrollo: ', style={'color': COLORES['coral_rosado']}),
                                    'Países con estructuras económicas menos diversificadas y '
                                    'mayor dependencia de sectores primarios. Requieren transformación estructural.'
                                ])
                            ], style={'lineHeight': '1.8'})
                        ], style={
                            'backgroundColor': 'white',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'border': f'2px solid {COLORES["purpura_frio"]}',
                            'marginBottom': '20px',
                            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                        }),
                        
                        # Hallazgo 3
                        html.Div([
                            html.H5('3. Migración de Países entre Clusters', className='mb-3', style={
                                'color': COLORES['azul_medio'],
                                'fontWeight': '600'
                            }),
                            html.P([
                                'El análisis de migración reveló que una proporción significativa de países ',
                                html.Strong('cambió de cluster entre 2007 y 2022', style={'color': COLORES['turquesa']}),
                                ', reflejando el impacto de eventos macroeconómicos globales. La crisis financiera de 2008-2009, la crisis de deuda '
                                'europea, las fluctuaciones en commodities y la pandemia de COVID-19 han reconfigurado posiciones relativas. '
                                'Algunos países emergentes avanzaron hacia perfiles más desarrollados, mientras que otros experimentaron retrocesos '
                                'debido a crisis de deuda o choques económicos.'
                            ], style={'textAlign': 'justify', 'lineHeight': '1.8'})
                        ], style={
                            'backgroundColor': 'white',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'border': f'2px solid {COLORES["magenta_suave"]}',
                            'marginBottom': '20px',
                            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                        }),
                        
                        # Hallazgo 4
                        html.Div([
                            html.H5('4. Validación del Modelo', className='mb-3', style={
                                'color': COLORES['azul_medio'],
                                'fontWeight': '600'
                            }),
                            html.P([
                                'Las ',
                                html.Strong('métricas de evaluación', style={'color': COLORES['turquesa']}),
                                ' (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score) confirmaron '
                                'la calidad de los clusters identificados. El método del codo sugirió entre 4 y 6 clusters como óptimos, '
                                'balanceando parsimonia y capacidad explicativa. La coherencia económica de los clusters validó la utilidad '
                                'del enfoque no supervisado para identificar patrones estructurales en datos macroeconómicos.'
                            ], style={'textAlign': 'justify', 'lineHeight': '1.8'})
                        ], style={
                            'backgroundColor': 'white',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'border': f'2px solid {COLORES["coral_rosado"]}',
                            'marginBottom': '30px',
                            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                        }),
                        
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '30px',
                        'borderRadius': '10px',
                        'border': f'1px solid {COLORES["azul_medio"]}',
                        'marginBottom': '30px'
                    }),
                    
                    # Relevancia de los Resultados
                    html.Div([
                        html.H4('Relevancia de los Resultados', className='mb-4', style={
                            'color': COLORES['azul_profundo'],
                            'fontWeight': '600'
                        }),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H5('Política Económica y Cooperación Internacional', className='mb-3', style={
                                        'color': COLORES['azul_medio'],
                                        'fontWeight': '600'
                                    }),
                                    html.P([
                                        'Los arquetipos identificados pueden informar el diseño de ',
                                        html.Strong('políticas económicas diferenciadas', style={'color': COLORES['turquesa']}),
                                        ' y mecanismos de '
                                        'cooperación internacional. Organismos como el FMI, Banco Mundial y bancos regionales de desarrollo pueden '
                                        'utilizar estos perfiles para calibrar programas de asistencia, condicionalidades y recomendaciones de política '
                                        'adaptadas a las características específicas de cada grupo.'
                                    ], style={'textAlign': 'justify', 'lineHeight': '1.8'})
                                ], style={
                                    'backgroundColor': 'white',
                                    'padding': '20px',
                                    'borderRadius': '8px',
                                    'border': f'2px solid {COLORES["azul_medio"]}',
                                    'height': '100%'
                                })
                            ], md=4, className='mb-3'),
                            
                            dbc.Col([
                                html.Div([
                                    html.H5('Gestión de Riesgos y Análisis de Vulnerabilidades', className='mb-3', style={
                                        'color': COLORES['azul_medio'],
                                        'fontWeight': '600'
                                    }),
                                    html.P([
                                        'La identificación de clusters con ',
                                        html.Strong('alta deuda externa o vulnerabilidad macroeconómica', style={'color': COLORES['coral_rosado']}),
                                        ' permite sistemas de '
                                        'alerta temprana y monitoreo de riesgos sistémicos. Los inversores internacionales, agencias de calificación '
                                        'crediticia y gestores de portafolio pueden utilizar estos perfiles para evaluar riesgos soberanos y ajustar '
                                        'estrategias de asignación de activos.'
                                    ], style={'textAlign': 'justify', 'lineHeight': '1.8'})
                                ], style={
                                    'backgroundColor': 'white',
                                    'padding': '20px',
                                    'borderRadius': '8px',
                                    'border': f'2px solid {COLORES["purpura_frio"]}',
                                    'height': '100%'
                                })
                            ], md=4, className='mb-3'),
                            
                            dbc.Col([
                                html.Div([
                                    html.H5('Investigación Económica', className='mb-3', style={
                                        'color': COLORES['azul_medio'],
                                        'fontWeight': '600'
                                    }),
                                    html.P([
                                        'Este trabajo demuestra el potencial del ',
                                        html.Strong('machine learning no supervisado', style={'color': COLORES['magenta_suave']}),
                                        ' para descubrir patrones en datos '
                                        'económicos complejos. Los clusters identificados pueden servir como variables categóricas en estudios '
                                        'econométricos posteriores, análisis de convergencia/divergencia económica y estudios comparativos internacionales.'
                                    ], style={'textAlign': 'justify', 'lineHeight': '1.8'})
                                ], style={
                                    'backgroundColor': 'white',
                                    'padding': '20px',
                                    'borderRadius': '8px',
                                    'border': f'2px solid {COLORES["magenta_suave"]}',
                                    'height': '100%'
                                })
                            ], md=4, className='mb-3')
                        ])
                    ], style={
                        'marginBottom': '30px'
                    }),
                    
                    # Aplicaciones Futuras
                    html.Div([
                        html.H4('Aplicaciones Futuras y Recomendaciones', className='mb-4', style={
                            'color': COLORES['azul_profundo'],
                            'fontWeight': '600'
                        }),
                        
                        # Extensiones Metodológicas
                        html.Div([
                            html.H5('Extensiones Metodológicas', className='mb-3', style={
                                'color': COLORES['azul_medio'],
                                'fontWeight': '600'
                            }),
                            html.Ul([
                                html.Li([
                                    html.Strong('Análisis temporal continuo: ', style={'color': COLORES['turquesa']}),
                                    'Extender el análisis a series temporales completas (2005-2024) para identificar trayectorias '
                                    'dinámicas y transiciones entre clusters a lo largo del tiempo.'
                                ], className='mb-2'),
                                html.Li([
                                    html.Strong('Métodos de clustering alternativos: ', style={'color': COLORES['turquesa']}),
                                    'Comparar K-Means con DBSCAN, clustering jerárquico para validar robustez de resultados.'
                                ], className='mb-2'),
                                html.Li([
                                    html.Strong('Reducción de dimensionalidad: ', style={'color': COLORES['turquesa']}),
                                    'Aplicar PCA, t-SNE o UMAP para visualizar clusters en espacios de menor dimensión y facilitar interpretación.'
                                ])
                            ], style={'lineHeight': '1.8'})
                        ], style={
                            'backgroundColor': 'white',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'border': f'2px solid {COLORES["azul_medio"]}',
                            'marginBottom': '20px',
                            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                        }),
                        
                        # Variables Adicionales
                        html.Div([
                            html.H5('Incorporación de Variables Adicionales', className='mb-3', style={
                                'color': COLORES['azul_medio'],
                                'fontWeight': '600'
                            }),
                            html.Ul([
                                html.Li('Indicadores de desarrollo humano (IDH, educación, salud)', className='mb-1'),
                                html.Li('Métricas de gobernanza y calidad institucional', className='mb-1'),
                                html.Li('Indicadores de innovación tecnológica y digitalización', className='mb-1'),
                                html.Li('Variables ambientales y de sostenibilidad (emisiones, energía renovable)', className='mb-1'),
                                html.Li('Indicadores de desigualdad (Gini, pobreza, distribución del ingreso)')
                            ], style={'lineHeight': '1.8'})
                        ], style={
                            'backgroundColor': 'white',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'border': f'2px solid {COLORES["purpura_frio"]}',
                            'marginBottom': '20px',
                            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                        }),
                        
                        # Modelos Predictivos
                        html.Div([
                            html.H5('Modelos Predictivos', className='mb-3', style={
                                'color': COLORES['azul_medio'],
                                'fontWeight': '600'
                            }),
                            html.P([
                                'Desarrollar ',
                                html.Strong('modelos supervisados', style={'color': COLORES['turquesa']}),
                                ' (Random Forest, Gradient Boosting, redes neuronales) para predecir '
                                'transiciones entre clusters y identificar factores determinantes de movilidad económica. Implementar '
                                'técnicas de análisis causal (regresión discontinua, variables instrumentales, diferencias en diferencias) '
                                'para cuantificar impactos de políticas específicas.'
                            ], style={'textAlign': 'justify', 'lineHeight': '1.8'})
                        ], style={
                            'backgroundColor': 'white',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'border': f'2px solid {COLORES["magenta_suave"]}',
                            'marginBottom': '20px',
                            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                        }),
                        
                        # Plataforma Interactiva
                        html.Div([
                            html.H5('Plataforma Interactiva', className='mb-3', style={
                                'color': COLORES['azul_medio'],
                                'fontWeight': '600'
                            }),
                            html.P([
                                'Expandir este dashboard con ',
                                html.Strong('funcionalidades avanzadas', style={'color': COLORES['turquesa']}),
                                ': descarga de datos procesados, comparación '
                                'personalizada de países, simulación de escenarios económicos, integración con APIs del Banco Mundial '
                                'para actualizaciones automáticas, y módulos educativos sobre clustering y análisis económico.'
                            ], style={'textAlign': 'justify', 'lineHeight': '1.8'})
                        ], style={
                            'backgroundColor': 'white',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'border': f'2px solid {COLORES["coral_rosado"]}',
                            'marginBottom': '20px',
                            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                        })
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '30px',
                        'borderRadius': '10px',
                        'border': f'1px solid {COLORES["azul_medio"]}'
                    })
                    
                ], lg=10, md=12, className='mx-auto')
            ])
        ], className='py-4')
    ])
]


app.layout = dbc.Container([
    # Header profesional con paleta de colores
    html.Div([
        html.Div([
            html.H1("Arquetipos Económicos Globales", 
                   className="text-white mb-2",
                   style={'fontWeight': '700', 'letterSpacing': '1px'}),
            html.H4("Análisis Comparativo Pre-Crisis (2007) vs Post-Pandemia (2022)",
                   className="text-white-50 mb-0",
                   style={'fontWeight': '300'})
        ], className="text-center py-4")
    ], style={
        'background': f'linear-gradient(135deg, {COLORES["azul_profundo"]} 0%, {COLORES["azul_medio"]} 50%, {COLORES["purpura_frio"]} 100%)',
        'marginBottom': '30px',
        'borderRadius': '0 0 15px 15px',
        'boxShadow': '0 4px 15px rgba(0,0,0,0.2)'
    }),
    
    dcc.Tabs(tabs, colors={
        'border': COLORES['azul_medio'],
        'primary': COLORES['turquesa'],
        'background': '#f8f9fa'
    })
], fluid=True, style={'maxWidth': '1400px', 'padding': '0 20px'})


# ==================== CALLBACKS PARA EDA 1 ====================

@app.callback(
    [Output('eda1-stats-table', 'children'),
     Output('eda1-histogram', 'figure'),
     Output('eda1-boxplot', 'figure')],
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
    
    return stats_table, fig_hist, fig_box


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
    Input('elbow-chart', 'id')  # Trigger automático
)
def update_elbow_chart(_):
    """Genera gráfico del método del codo con k=4 marcado"""
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
    
    # Marcar k=4 como óptimo
    selected_k = 4
    idx = list(K_range).index(selected_k)
    fig.add_trace(go.Scatter(
        x=[selected_k],
        y=[inertias[idx]],
        mode='markers',
        marker=dict(size=20, color=COLORES['coral_rosado'], 
                   line=dict(color='white', width=3)),
        name=f'k={selected_k} (Óptimo)'
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
    [Output('metrics-table', 'children'),
     Output('cluster-dist-2007', 'figure'),
     Output('cluster-dist-2022', 'figure')],
    Input('metrics-table', 'id')  # Trigger automático
)
def update_evaluation_tab(_):
    """Actualiza la pestaña de evaluación después del entrenamiento"""
    if kmeans_2007 is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return html.Div("Entrena los modelos primero"), empty_fig, empty_fig
    
    # Calcular métricas usando datos PCA
    metrics_2007 = calculate_metrics(X_2007_pca, df_2007_clustered['Cluster'])
    metrics_2022 = calculate_metrics(X_2022_pca, df_2022_clustered['Cluster'])
    
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
    Input('choropleth-2007', 'id')  # Trigger automático
)
def update_choropleth_maps(_):
    """Genera mapas de coropletas con los clusters"""
    if df_2007_clustered is None or df_2022_clustered is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return empty_fig, empty_fig
    
    # Añadir nombres de arquetipos a los dataframes
    df_2007_map = df_2007_clustered.copy()
    df_2007_map['Arquetipo'] = df_2007_map['Cluster'].apply(
        lambda x: get_cluster_archetype(x, 2007, df_2007_clustered)[0]
    )
    
    df_2022_map = df_2022_clustered.copy()
    df_2022_map['Arquetipo'] = df_2022_map['Cluster'].apply(
        lambda x: get_cluster_archetype(x, 2022, df_2022_clustered)[0]
    )
    
    # Mapa de colores consistente
    color_map = {
        "Economías Desarrolladas": COLORES['turquesa'],
        "Mercados Emergentes": COLORES['magenta_suave'],
        "Economías en Transición": COLORES['azul_medio'],
        "Economías en Desarrollo": COLORES['coral_rosado']
    }
    
    # Mapa 2007
    fig_2007 = px.choropleth(
        df_2007_map,
        locations='Country Code',
        color='Arquetipo',
        hover_name='Country Name',
        hover_data={'Cluster': True, 'gdp_per_capita': ':.0f'},
        color_discrete_map=color_map,
        category_orders={'Arquetipo': ['Economías Desarrolladas', 'Mercados Emergentes', 'Economías en Transición', 'Economías en Desarrollo']},
        title='Distribución Global de Arquetipos Económicos - 2007'
    )
    fig_2007.update_layout(template='plotly_white', height=500)
    
    # Mapa 2022
    fig_2022 = px.choropleth(
        df_2022_map,
        locations='Country Code',
        color='Arquetipo',
        hover_name='Country Name',
        hover_data={'Cluster': True, 'gdp_per_capita': ':.0f'},
        color_discrete_map=color_map,
        category_orders={'Arquetipo': ['Economías Desarrolladas', 'Mercados Emergentes', 'Economías en Transición', 'Economías en Desarrollo']},
        title='Distribución Global de Arquetipos Económicos - 2022'
    )
    fig_2022.update_layout(template='plotly_white', height=500)
    
    return fig_2007, fig_2022


@app.callback(
    Output('centroid-heatmap', 'figure'),
    Input('centroid-year-dropdown', 'value')
)
def update_centroid_heatmap(year):
    """Genera heatmap de centroides"""
    if kmeans_2007 is None or kmeans_2022 is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return empty_fig
    
    kmeans = kmeans_2007 if year == 2007 else kmeans_2022
    scaler = scaler_2007 if year == 2007 else scaler_2022
    pca = pca_2007 if year == 2007 else pca_2022
    
    # Los centroides pueden estar en espacio PCA o directo según la configuración
    centroids_scaled = kmeans.cluster_centers_
    
    # Si se usó PCA, transformar de vuelta
    if pca is not None:
        centroids_scaled = pca.inverse_transform(centroids_scaled)
    
    # Transformar de escala normalizada a valores originales
    centroids = scaler.inverse_transform(centroids_scaled)
    
    # Crear heatmap (usar key_variables que son las que se usaron en clustering)
    fig = go.Figure(data=go.Heatmap(
        z=centroids.T,
        x=[f'Cluster {i}' for i in range(optimal_k)],
        y=[var.replace('_', ' ').title() for var in key_variables],
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
    [Input('scatter3d-x', 'value'),
     Input('scatter3d-y', 'value'),
     Input('scatter3d-z', 'value')]
)
def update_scatter3d(var_x, var_y, var_z):
    """Genera scatter plots 3D con clusters"""
    if df_2007_clustered is None or df_2022_clustered is None or not all([var_x, var_y, var_z]):
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
    Input('metrics-comparison', 'id')  # Trigger automático
)
def update_metrics_indicators(_):
    """Actualiza los indicadores y comparaciones de métricas"""
    if kmeans_2007 is None or df_2007_clustered is None or df_2022_clustered is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return empty_fig, html.Div("Entrena los modelos primero"), empty_fig
    
    # Usar los datos PCA transformados que ya tenemos
    try:
        # Calcular métricas usando los datos PCA que se usaron para clustering
        metrics_2007 = calculate_metrics(X_2007_pca, df_2007_clustered['Cluster'])
    
        metrics_2022 = calculate_metrics(X_2022_pca, df_2022_clustered['Cluster'])
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error al calcular métricas: {str(e)}")
        return empty_fig, html.Div(f"Error: {str(e)}"), empty_fig
    
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


# ==================== CALLBACKS PARA INTERPRETACIÓN DE CLUSTERS ====================

@app.callback(
    Output('interp-cluster-dropdown', 'options'),
    Input('interp-cluster-dropdown', 'id')  # Trigger automático
)
def update_cluster_options(_):
    """Actualiza las opciones de clusters disponibles"""
    if optimal_k is None:
        return []
    return [{'label': f'Cluster {i}', 'value': i} for i in range(optimal_k)]


@app.callback(
    [Output('cluster-summary-card', 'children'),
     Output('cluster-profile-radar', 'figure'),
     Output('cluster-profile-bars', 'figure'),
     Output('cluster-countries-list', 'children')],
    [Input('interp-year-dropdown', 'value'),
     Input('interp-cluster-dropdown', 'value')]
)
def update_cluster_interpretation(year, cluster_id):
    """Genera la interpretación completa del cluster seleccionado"""
    if df_2007_clustered is None or df_2022_clustered is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return html.Div("Entrena los modelos primero"), empty_fig, empty_fig, html.Div()
    
    # Seleccionar datos
    df_clustered = df_2007_clustered if year == 2007 else df_2022_clustered
    kmeans = kmeans_2007 if year == 2007 else kmeans_2022
    scaler = scaler_2007 if year == 2007 else scaler_2022
    pca = pca_2007 if year == 2007 else pca_2022
    
    # Filtrar países del cluster
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
    num_countries = len(cluster_data)
    
    # Obtener centroide
    centroid_scaled = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
    
    # Si se usó PCA, transformar de vuelta
    if pca is not None:
        centroid_scaled = pca.inverse_transform(centroid_scaled)
    
    # Transformar de escala normalizada a valores originales
    centroid_original = scaler.inverse_transform(centroid_scaled)
    centroid = pd.Series(centroid_original[0], index=key_variables)
    
    # Crear diccionario de características
    cluster_profile = centroid.to_dict()
    
    # Obtener arquetipo, color y descripción usando la función centralizada
    arquetipo, color, descripcion = get_cluster_archetype(cluster_id, year, df_clustered)
    
    # Obtener valores clave
    gdp_per_capita = cluster_profile.get('gdp_per_capita', 0)
    external_debt_gni = cluster_profile.get('external_debt_gni', 0)
    
    # Tarjeta de resumen
    summary_card = dbc.Card([
        dbc.CardHeader([
            html.H5(f'Cluster {cluster_id}: {arquetipo}', className='mb-0')
        ], style={'backgroundColor': color, 'color': 'white'}),
        dbc.CardBody([
            html.P(descripcion, className='mb-3'),
            dbc.Row([
                dbc.Col([
                    html.H6('Número de Países:', className='text-muted'),
                    html.H4(f'{num_countries}', className='mb-0')
                ], width=4),
                dbc.Col([
                    html.H6('PIB per cápita promedio:', className='text-muted'),
                    html.H4(f'${gdp_per_capita:,.0f}', className='mb-0')
                ], width=4),
                dbc.Col([
                    html.H6('Deuda externa (% INB):', className='text-muted'),
                    html.H4(f'{external_debt_gni:.1f}%', className='mb-0')
                ], width=4)
            ])
        ])
    ], className='mb-4')
    
    # Gráfico radar del perfil
    fig_radar = go.Figure()
    
    # Normalizar valores para el radar (0-1)
    centroid_array = np.array([cluster_profile[var] for var in key_variables])
    centroid_df = pd.DataFrame(centroid_array.reshape(1, -1), columns=key_variables)
    centroid_normalized = scaler.transform(centroid_df)[0]
    centroid_normalized = (centroid_normalized - centroid_normalized.min()) / (centroid_normalized.max() - centroid_normalized.min() + 0.001)
    
    fig_radar.add_trace(go.Scatterpolar(
        r=centroid_normalized,
        theta=[var.replace('_', ' ').title() for var in key_variables],
        fill='toself',
        name=f'Cluster {cluster_id}',
        line_color=color,
        fillcolor=color,
        opacity=0.6
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title=f'Perfil Radar del Cluster {cluster_id}',
        template='plotly_white',
        height=500
    )
    
    # Gráfico de barras comparativo con la media global
    global_mean = df_clustered[key_variables].mean()
    
    comparison_data = []
    for var in key_variables[:8]:  # Top 8 variables
        comparison_data.append({
            'Variable': var.replace('_', ' ').title(),
            'Cluster': cluster_profile[var],
            'Media Global': global_mean[var]
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(
        name=f'Cluster {cluster_id}',
        x=df_comparison['Variable'],
        y=df_comparison['Cluster'],
        marker_color=color
    ))
    fig_bars.add_trace(go.Bar(
        name='Media Global',
        x=df_comparison['Variable'],
        y=df_comparison['Media Global'],
        marker_color=COLORES['azul_profundo'],
        opacity=0.6
    ))
    
    fig_bars.update_layout(
        title=f'Comparación con Media Global',
        xaxis_title='Variable',
        yaxis_title='Valor',
        barmode='group',
        template='plotly_white',
        height=500,
        xaxis={'tickangle': 45}
    )
    
    # Lista de países
    countries_sorted = cluster_data.sort_values('gdp_per_capita', ascending=False)
    
    countries_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th('#'),
            html.Th('País'),
            html.Th('Código'),
            html.Th('PIB per cápita'),
            html.Th('Crecimiento PIB (%)'),
            html.Th('Inflación (%)')
        ])),
        html.Tbody([
            html.Tr([
                html.Td(i+1),
                html.Td(row['Country Name']),
                html.Td(row['Country Code']),
                html.Td(f"${row.get('gdp_per_capita', 0):,.0f}"),
                html.Td(f"{row.get('gdp_growth', 0):.2f}"),
                html.Td(f"{row.get('inflation', 0):.2f}")
            ]) for i, (_, row) in enumerate(countries_sorted.head(20).iterrows())
        ])
    ], bordered=True, striped=True, hover=True, responsive=True, size='sm')
    
    return summary_card, fig_radar, fig_bars, countries_table


@app.callback(
    Output('cluster-comparison-box', 'figure'),
    [Input('interp-year-dropdown', 'value'),
     Input('comparison-variable-dropdown', 'value')]
)
def update_cluster_comparison(year, variable):
    """Genera boxplot comparativo entre clusters"""
    if df_2007_clustered is None or df_2022_clustered is None or not variable:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return empty_fig
    
    df_clustered = df_2007_clustered if year == 2007 else df_2022_clustered
    
    fig = go.Figure()
    
    for cluster_id in range(optimal_k):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id][variable]
        fig.add_trace(go.Box(
            y=cluster_data,
            name=f'Cluster {cluster_id}',
            marker_color=PALETTE[cluster_id % len(PALETTE)]
        ))
    
    fig.update_layout(
        title=f'Comparación de {variable.replace("_", " ").title()} entre Clusters ({year})',
        yaxis_title=variable.replace('_', ' ').title(),
        xaxis_title='Cluster',
        template='plotly_white',
        showlegend=True,
        height=450
    )
    
    return fig


@app.callback(
    [Output('migration-analysis', 'children'),
     Output('sankey-migration', 'figure')],
    Input('migration-analysis', 'id')  # Trigger automático
)
def update_migration_analysis(_):
    """Analiza la migración de países entre clusters de 2007 a 2022"""
    if df_2007_clustered is None or df_2022_clustered is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Entrena los modelos primero")
        return html.Div("Entrena los modelos primero"), empty_fig
    
    # Crear tabla de migración
    df_migration = pd.merge(
        df_2007_clustered[['Country Code', 'Country Name', 'Cluster']],
        df_2022_clustered[['Country Code', 'Cluster']],
        on='Country Code',
        suffixes=('_2007', '_2022')
    )
    
    # Calcular estadísticas de migración
    total_countries = len(df_migration)
    stable_countries = len(df_migration[df_migration['Cluster_2007'] == df_migration['Cluster_2022']])
    migrated_countries = total_countries - stable_countries
    stability_rate = (stable_countries / total_countries) * 100
    
    # Análisis por cluster
    migration_summary = []
    for cluster_2007 in range(optimal_k):
        cluster_2007_countries = df_migration[df_migration['Cluster_2007'] == cluster_2007]
        for cluster_2022 in range(optimal_k):
            count = len(cluster_2007_countries[cluster_2007_countries['Cluster_2022'] == cluster_2022])
            if count > 0:
                migration_summary.append({
                    'from': cluster_2007,
                    'to': cluster_2022,
                    'count': count
                })
    
    # Tarjeta de resumen
    summary_card = dbc.Card([
        dbc.CardHeader('Resumen de Migración 2007 → 2022', className='fw-bold'),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6('Total de Países:', className='text-muted'),
                    html.H4(f'{total_countries}', className='mb-0')
                ], width=3),
                dbc.Col([
                    html.H6('Países Estables:', className='text-muted'),
                    html.H4(f'{stable_countries}', className='mb-0', style={'color': COLORES['turquesa']})
                ], width=3),
                dbc.Col([
                    html.H6('Países Migrados:', className='text-muted'),
                    html.H4(f'{migrated_countries}', className='mb-0', style={'color': COLORES['coral_rosado']})
                ], width=3),
                dbc.Col([
                    html.H6('Tasa de Estabilidad:', className='text-muted'),
                    html.H4(f'{stability_rate:.1f}%', className='mb-0')
                ], width=3)
            ])
        ])
    ], className='mb-4')
    
    # Crear diagrama de Sankey
    source = []
    target = []
    value = []
    labels = [f'2007: Cluster {i}' for i in range(optimal_k)] + [f'2022: Cluster {i}' for i in range(optimal_k)]
    
    for item in migration_summary:
        source.append(item['from'])
        target.append(item['to'] + optimal_k)
        value.append(item['count'])
    
    # Colores para el Sankey
    node_colors = PALETTE[:optimal_k] + PALETTE[:optimal_k]
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color='rgba(0,0,0,0.2)'
        )
    )])
    
    fig_sankey.update_layout(
        title='Flujo de Países entre Clusters (2007 → 2022)',
        font=dict(size=12),
        height=600,
        template='plotly_white'
    )
    
    return summary_card, fig_sankey


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host="0.0.0.0", port=port)