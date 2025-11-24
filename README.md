# Dashboard de Arquetipos de la Economía Global

## Descripción del Proyecto

Este proyecto universitario desarrolla un dashboard web interactivo que analiza la estructura de la economía global mediante técnicas de machine learning no supervisado. Utiliza clustering K-Means para identificar arquetipos económicos y compara dos períodos críticos de la historia económica reciente:

- **Año 2007:** El mundo pre-crisis financiera global
- **Año 2022:** El mundo post-pandemia y de tensiones geopolíticas

El análisis se basa en 14 indicadores macroeconómicos del Banco Mundial (World Development Indicators) para más de 200 países, permitiendo identificar patrones estructurales, evaluar transformaciones económicas y visualizar migraciones de países entre diferentes perfiles económicos.

## Pregunta de Investigación

¿Cómo han cambiado los perfiles económicos de los países y su agrupación global, al comparar la estructura pre-crisis financiera (año 2007) con el panorama post-pandemia (año 2022), utilizando un modelo de clustering K-Means?

## Características Principales

### Análisis Exploratorio de Datos (EDA)
- Estadísticas descriptivas interactivas por año y variable
- Histogramas y boxplots para análisis de distribuciones
- Matrices de correlación entre indicadores económicos
- Gráficos de dispersión comparativos (2007 vs 2022)

### Metodología de Clustering
- Método del Codo (Elbow Method) para selección óptima de k
- Entrenamiento interactivo de modelos K-Means
- Evaluación con múltiples métricas (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score)
- Análisis de distribución de clusters

### Visualización de Resultados
- Mapas coropléticos interactivos por año
- Heatmaps de centroides de clusters
- Gráficos 3D de dispersión con variables personalizables
- Comparación de métricas de evaluación

### Interpretación de Clusters
- Clasificación automática de arquetipos económicos
- Perfiles detallados por cluster (radar charts, estadísticas)
- Análisis de países por cluster
- Diagrama de Sankey de migración entre clusters (2007 → 2022)
- Cálculo de tasa de estabilidad económica

## Stack Tecnológico

### Lenguajes y Frameworks
- **Python 3.11+**
- **Dash** - Framework para aplicaciones web interactivas
- **Plotly** - Visualizaciones interactivas
- **Dash Bootstrap Components** - Componentes de interfaz

### Librerías de Análisis
- **pandas** - Manipulación de datos
- **scikit-learn** - Machine learning (K-Means, StandardScaler)
- **numpy** - Computación numérica
- **statsmodels** - Análisis estadístico

### Base de Datos
- **PostgreSQL** - Base de datos relacional (alojada en Render)
- **SQLAlchemy** - ORM y conexión a base de datos
- **psycopg2-binary** - Adaptador PostgreSQL para Python

### Despliegue
- **Docker** - Containerización
- **Gunicorn** - Servidor WSGI de producción
- **Render** - Plataforma de despliegue en la nube

## Estructura del Proyecto

```
Dataviz_DashFinal/
│
├── dashboard.py              # Aplicación principal de Dash
├── consultas_postgres.py     # Módulo de acceso a base de datos
├── cargar_postgres.py        # Script ETL para carga inicial de datos
│
├── data/                     # Datos fuente
│   ├── data.csv              # World Bank Development Indicators
│   └── Series - Metadata.csv # Metadatos de series
│
├── requirements.txt          # Dependencias de Python
├── Dockerfile                # Configuración de contenedor Docker
├── .env                      # Variables de entorno (no incluido en repo)
└── README.md                 # Documentación del proyecto
```

## Instalación y Configuración

### Requisitos Previos
- Python 3.11 o superior
- PostgreSQL (local o remoto)
- Docker (opcional, para containerización)

### Instalación Local

1. **Clonar el repositorio**
```bash
git clone https://github.com/LasciaStare/Dataviz_DashFinal.git
cd Dataviz_DashFinal
```

2. **Crear entorno virtual**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**

Crear archivo `.env` en la raíz del proyecto:
```env
DB_URL=postgresql://usuario:contraseña@host:puerto/nombre_db
```

5. **Cargar datos en PostgreSQL**
```bash
python cargar_postgres.py
```

6. **Ejecutar la aplicación**
```bash
python dashboard.py
```

La aplicación estará disponible en `http://localhost:8050`

### Despliegue con Docker

1. **Construir la imagen**
```bash
docker build -t dashboard-economia .
```

2. **Ejecutar el contenedor**
```bash
docker run -p 8050:8050 --env-file .env dashboard-economia
```

### Despliegue en Render

1. **Crear base de datos PostgreSQL en Render**
   - Crear una instancia de PostgreSQL en Render
   - Copiar la URL de conexión interna

2. **Configurar variables de entorno en Render**
   - `DB_URL`: URL de conexión a PostgreSQL

3. **Desplegar desde repositorio GitHub**
   - Conectar el repositorio a Render
   - Seleccionar Docker como método de despliegue
   - El Dockerfile se encargará de la configuración

## Uso del Dashboard

### 1. Introducción y Contexto
Navega por las pestañas iniciales para comprender el contexto del proyecto, el planteamiento del problema y los objetivos.

### 2. Análisis Exploratorio (EDA)
- Selecciona año (2007 o 2022) y variable económica
- Explora estadísticas descriptivas, distribuciones y correlaciones
- Compara patrones entre ambos períodos

### 3. Metodología
- **Definición:** Conoce el enfoque de clustering aplicado
- **Preparación:** Revisa los pasos de preprocesamiento de datos
- **Selección:** Usa el método del codo para elegir k óptimo (slider 2-8)
- **Evaluación:** Haz clic en "Entrenar Modelos" y revisa métricas

### 4. Resultados
- **Visualización:** Explora mapas mundiales, heatmaps y gráficos 3D
- **Indicadores:** Compara métricas de evaluación entre años
- **Interpretación:** Analiza arquetipos, perfiles de clusters y migraciones de países

### 5. Conclusiones
Revisa los hallazgos principales, relevancia de resultados y aplicaciones futuras.

## Indicadores Económicos Utilizados

1. **GDP per capita (current US$)** - PIB per cápita
2. **GDP growth (annual %)** - Crecimiento del PIB
3. **Inflation, GDP deflator (annual %)** - Inflación
4. **Foreign direct investment, net inflows (% of GDP)** - Inversión extranjera directa
5. **Gross fixed capital formation (% of GDP)** - Formación bruta de capital fijo
6. **External debt stocks, total (DOD, current US$)** - Deuda externa total
7. **Present value of external debt (current US$)** - Valor presente de deuda externa
8. **External debt (% of GNI)** - Deuda externa como % del INB
9. **Debt service (% of exports)** - Servicio de deuda
10. **Total reserves in months of imports** - Reservas internacionales
11. **Current account balance (% of GDP)** - Balanza de cuenta corriente
12. **Unemployment, total (% of labor force)** - Desempleo
13. **Trade (% of GDP)** - Comercio total
14. **Population, total** - Población total

## Fuente de Datos

**World Bank - World Development Indicators (WDI)**
- Período: 2005-2024
- Cobertura: 200+ países
- Actualización: Anual
- Fuente: https://data.worldbank.org/

## Metodología de Machine Learning

### Preprocesamiento
1. Filtrado de datos por año (2007 y 2022)
2. Tratamiento de valores faltantes (imputación con mediana)
3. Detección y tratamiento de outliers (método IQR, multiplicador 3x)
4. Eliminación de variables con >50% de datos faltantes
5. Normalización con StandardScaler (z-score)

### Modelo
- **Algoritmo:** K-Means clustering
- **Inicialización:** k-means++
- **Máximo de iteraciones:** 300
- **Random state:** 42 (reproducibilidad)

### Evaluación
- **Silhouette Score:** Mide cohesión y separación de clusters (0 a 1, mayor es mejor)
- **Davies-Bouldin Index:** Ratio de dispersión intra-cluster vs inter-cluster (menor es mejor)
- **Calinski-Harabasz Score:** Ratio de varianza between/within clusters (mayor es mejor)
- **Inertia (WCSS):** Suma de distancias cuadradas intra-cluster (menor es mejor)

## Limitaciones

1. **Sensibilidad a outliers:** K-Means puede ser afectado por valores extremos
2. **Supuesto de clusters esféricos:** Puede no capturar estructuras económicas complejas
3. **Número fijo de clusters:** No identifica automáticamente el número óptimo
4. **Sensibilidad a inicialización:** Aunque k-means++ mitiga este problema
5. **Datos faltantes:** Algunos países tienen información incompleta
6. **Análisis estático:** Solo compara dos años específicos, no muestra evolución continua

## Mejoras Futuras

1. Análisis temporal continuo (2005-2024)
2. Algoritmos de clustering alternativos (DBSCAN, Hierarchical, GMM)
3. Técnicas de reducción de dimensionalidad (PCA, t-SNE, UMAP)
4. Incorporación de variables adicionales (IDH, gobernanza, innovación)
5. Modelos predictivos para transiciones entre clusters
6. Análisis causal de factores determinantes
7. Validación externa con clasificaciones establecidas
8. API para actualizaciones automáticas de datos


## Licencia

Este proyecto es de uso académico y educativo.

