"""
Script para cargar y preprocesar datos del Banco Mundial y subirlos a PostgreSQL.

Este script:
1. Carga los datos económicos del Banco Mundial desde CSV
2. Reestructura los datos del formato wide a long
3. Filtra los datos para los años 2007 y 2022
4. Aplica preprocesamiento profesional: limpieza, imputación, eliminación de outliers
5. Sube los datasets procesados a PostgreSQL en tablas separadas

Autor: Proyecto Dashboard Arquetipos Economía Global
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings('ignore')

# Cargar variables de entorno
load_dotenv()

# Constantes
DATA_PATH = 'data/data.csv'
METADATA_PATH = 'data/Series - Metadata.csv'
TARGET_YEARS = [2007, 2022]

# Indicadores económicos a utilizar (según el proyecto)
INDICATORS = {
    'NY.GDP.PCAP.CD': 'gdp_per_capita',
    'NY.GDP.MKTP.KD.ZG': 'gdp_growth',
    'NY.GDP.DEFL.KD.ZG': 'inflation',
    'BX.KLT.DINV.WD.GD.ZS': 'fdi_inflows',
    'NE.GDI.FTOT.ZS': 'gross_fixed_capital',
    'DT.DOD.DECT.CD': 'external_debt_total',
    'DT.DOD.PVLX.CD': 'external_debt_present_value',
    'DT.DOD.DECT.GN.ZS': 'external_debt_gni',
    'DT.TDS.DPPF.XP.ZS': 'debt_service',
    'FI.RES.TOTL.MO': 'reserves_months_imports',
    'BN.CAB.XOKA.GD.ZS': 'current_account_balance',
    'SL.UEM.TOTL.ZS': 'unemployment',
    'NE.TRD.GNFS.ZS': 'trade_gdp',
    'SP.POP.TOTL': 'population'
}


def cargar_datos_raw():
    """
    Carga los datos crudos desde el archivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame con los datos crudos
    """
    print("Cargando datos desde CSV...")
    df = pd.read_csv(DATA_PATH)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def reestructurar_datos(df):
    """
    Reestructura los datos de formato wide (años en columnas) a formato long.
    
    Args:
        df (pd.DataFrame): DataFrame en formato wide
        
    Returns:
        pd.DataFrame: DataFrame en formato long
    """
    print("\nReestructurando datos de wide a long format...")
    
    # Identificar columnas de años
    year_cols = [col for col in df.columns if '[YR' in col]
    
    # Crear DataFrame long usando melt
    df_long = df.melt(
        id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
        value_vars=year_cols,
        var_name='year_col',
        value_name='value'
    )
    
    # Extraer el año de la columna
    df_long['year'] = df_long['year_col'].str.extract(r'(\d{4})').astype(int)
    df_long = df_long.drop('year_col', axis=1)
    
    # Limpiar valores no numéricos
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    
    print(f"Datos reestructurados: {df_long.shape[0]} filas")
    return df_long


def filtrar_por_años(df_long, years):
    """
    Filtra los datos para mantener solo los años especificados.
    
    Args:
        df_long (pd.DataFrame): DataFrame en formato long
        years (list): Lista de años a mantener
        
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    print(f"\nFiltrando datos para años: {years}")
    df_filtered = df_long[df_long['year'].isin(years)].copy()
    print(f"Datos filtrados: {df_filtered.shape[0]} filas")
    return df_filtered


def filtrar_por_indicadores(df_filtered):
    """
    Filtra los datos para mantener solo los indicadores relevantes.
    
    Args:
        df_filtered (pd.DataFrame): DataFrame filtrado por años
        
    Returns:
        pd.DataFrame: DataFrame con solo los indicadores relevantes
    """
    print(f"\nFiltrando {len(INDICATORS)} indicadores relevantes...")
    df_indicators = df_filtered[df_filtered['Series Code'].isin(INDICATORS.keys())].copy()
    
    # Mapear los códigos a nombres descriptivos
    df_indicators['indicator'] = df_indicators['Series Code'].map(INDICATORS)
    
    print(f"Datos con indicadores: {df_indicators.shape[0]} filas")
    return df_indicators


def pivotar_por_indicador(df_indicators):
    """
    Pivota los datos para tener un indicador por columna.
    
    Args:
        df_indicators (pd.DataFrame): DataFrame con indicadores
        
    Returns:
        dict: Diccionario con DataFrames por año
    """
    print("\nPivotando datos por indicador...")
    
    datasets_por_año = {}
    
    for year in TARGET_YEARS:
        df_year = df_indicators[df_indicators['year'] == year].copy()
        
        # Pivotar para tener indicadores como columnas
        df_pivot = df_year.pivot_table(
            index=['Country Name', 'Country Code'],
            columns='indicator',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        df_pivot['year'] = year
        datasets_por_año[year] = df_pivot
        
        print(f"Año {year}: {df_pivot.shape[0]} países, {df_pivot.shape[1]} columnas")
    
    return datasets_por_año


def limpiar_y_preprocesar(df, year):
    """
    Aplica preprocesamiento profesional a los datos:
    - Elimina países con demasiados valores faltantes
    - Imputa valores faltantes con la mediana
    - Detecta y maneja outliers extremos
    - Valida consistencia de datos
    
    Args:
        df (pd.DataFrame): DataFrame a limpiar
        year (int): Año de los datos
        
    Returns:
        pd.DataFrame: DataFrame limpio y preprocesado
    """
    print(f"\n{'='*60}")
    print(f"PREPROCESAMIENTO DE DATOS PARA AÑO {year}")
    print(f"{'='*60}")
    
    df_clean = df.copy()
    
    # 1. Estadísticas iniciales
    print(f"\n1. DATOS INICIALES:")
    print(f"   - Países: {len(df_clean)}")
    print(f"   - Variables: {len([col for col in df_clean.columns if col not in ['Country Name', 'Country Code', 'year']])}")
    
    # 2. Análisis de valores faltantes
    print(f"\n2. ANÁLISIS DE VALORES FALTANTES:")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    missing_info = []
    
    for col in numeric_cols:
        missing_count = df_clean[col].isna().sum()
        missing_pct = (missing_count / len(df_clean)) * 100
        missing_info.append({
            'variable': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })
    
    missing_df = pd.DataFrame(missing_info).sort_values('missing_pct', ascending=False)
    print(missing_df.to_string(index=False))
    
    # 3. Eliminar países con más del 50% de datos faltantes
    print(f"\n3. ELIMINACIÓN DE PAÍSES CON DATOS INSUFICIENTES:")
    threshold_missing = 0.5
    missing_per_country = df_clean[numeric_cols].isna().sum(axis=1) / len(numeric_cols)
    countries_to_drop = missing_per_country[missing_per_country > threshold_missing].index
    
    print(f"   - Países con >{threshold_missing*100}% de datos faltantes: {len(countries_to_drop)}")
    
    if len(countries_to_drop) > 0:
        dropped_countries = df_clean.loc[countries_to_drop, 'Country Name'].tolist()
        print(f"   - Países eliminados: {', '.join(dropped_countries[:10])}" + 
              (f" y {len(dropped_countries)-10} más" if len(dropped_countries) > 10 else ""))
        df_clean = df_clean.drop(countries_to_drop)
    
    # 4. Imputación de valores faltantes con la mediana
    print(f"\n4. IMPUTACIÓN DE VALORES FALTANTES:")
    for col in numeric_cols:
        missing_before = df_clean[col].isna().sum()
        if missing_before > 0:
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value, inplace=True)
            print(f"   - {col}: {missing_before} valores imputados con mediana = {median_value:.2f}")
    
    # 5. Detección y manejo de outliers extremos (IQR method)
    print(f"\n5. DETECCIÓN Y MANEJO DE OUTLIERS:")
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 3 IQR para outliers extremos
        upper_bound = Q3 + 3 * IQR
        
        outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        n_outliers = outliers_mask.sum()
        
        if n_outliers > 0:
            # Reemplazar outliers extremos con los límites
            df_clean.loc[outliers_mask & (df_clean[col] < lower_bound), col] = lower_bound
            df_clean.loc[outliers_mask & (df_clean[col] > upper_bound), col] = upper_bound
            print(f"   - {col}: {n_outliers} outliers ajustados")
    
    # 6. Validación de datos
    print(f"\n6. VALIDACIÓN DE DATOS:")
    
    # Validar que no haya valores negativos donde no deberían estar
    positive_cols = ['gdp_per_capita', 'population', 'trade_gdp']
    for col in positive_cols:
        if col in df_clean.columns:
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                print(f"   - ADVERTENCIA: {col} tiene {negative_count} valores negativos")
                # Convertir negativos a valor absoluto o eliminar
                df_clean = df_clean[df_clean[col] >= 0]
    
    # Validar rangos razonables para porcentajes
    percentage_cols = [col for col in df_clean.columns if any(
        keyword in col for keyword in ['pct', 'rate', 'unemployment', 'growth', 'inflation']
    )]
    
    for col in percentage_cols:
        if col in df_clean.columns:
            out_of_range = ((df_clean[col] < -100) | (df_clean[col] > 500)).sum()
            if out_of_range > 0:
                print(f"   - ADVERTENCIA: {col} tiene {out_of_range} valores fuera de rango esperado")
    
    # 7. Resumen final
    print(f"\n7. DATOS FINALES DESPUÉS DEL PREPROCESAMIENTO:")
    print(f"   - Países finales: {len(df_clean)}")
    print(f"   - Variables finales: {len(numeric_cols)}")
    print(f"   - Valores faltantes totales: {df_clean[numeric_cols].isna().sum().sum()}")
    print(f"   - Completitud de datos: {((1 - df_clean[numeric_cols].isna().sum().sum() / (len(df_clean) * len(numeric_cols))) * 100):.2f}%")
    
    return df_clean


def conectar_postgres():
    """
    Crea conexión a la base de datos PostgreSQL.
    
    Returns:
        sqlalchemy.engine.Engine: Motor de conexión a PostgreSQL
    """
    print("\nConectando a PostgreSQL...")
    
    # Obtener URL de conexión desde variables de entorno
    db_url = os.getenv('DB_URL')
    
    # Validar que la URL esté presente
    if not db_url:
        raise ValueError(
            "Falta la variable DB_URL en el archivo .env\n"
            "Asegúrate de definir: DB_URL=postgresql://user:password@host:port/database"
        )
    
    connection_string = db_url
    
    # Crear engine
    engine = create_engine(connection_string)
    
    # Probar conexión
    try:
        with engine.connect() as conn:
            print("Conexión exitosa a PostgreSQL")
    except Exception as e:
        print(f"Error al conectar a PostgreSQL: {e}")
        raise
    
    return engine


def subir_a_postgres(datasets_por_año, engine):
    """
    Sube los datasets a PostgreSQL en tablas separadas.
    
    Args:
        datasets_por_año (dict): Diccionario con DataFrames por año
        engine (sqlalchemy.engine.Engine): Motor de conexión
    """
    print("\n" + "="*60)
    print("SUBIENDO DATOS A POSTGRESQL")
    print("="*60)
    
    for year, df in datasets_por_año.items():
        table_name = f"paises_{year}"
        print(f"\nSubiendo tabla: {table_name}")
        print(f"  - Registros: {len(df)}")
        print(f"  - Columnas: {len(df.columns)}")
        
        try:
            # Subir a PostgreSQL (reemplazar si existe)
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            print(f"  ✓ Tabla {table_name} creada exitosamente")
            
        except Exception as e:
            print(f"  ✗ Error al crear tabla {table_name}: {e}")
            raise


def main():
    """
    Función principal que ejecuta todo el pipeline de carga y preprocesamiento.
    """
    print("\n" + "="*60)
    print("PIPELINE DE CARGA Y PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    try:
        # 1. Cargar datos crudos
        df_raw = cargar_datos_raw()
        
        # 2. Reestructurar datos
        df_long = reestructurar_datos(df_raw)
        
        # 3. Filtrar por años
        df_filtered = filtrar_por_años(df_long, TARGET_YEARS)
        
        # 4. Filtrar por indicadores
        df_indicators = filtrar_por_indicadores(df_filtered)
        
        # 5. Pivotar por indicador
        datasets_por_año = pivotar_por_indicador(df_indicators)
        
        # 6. Limpiar y preprocesar cada dataset
        for year in TARGET_YEARS:
            datasets_por_año[year] = limpiar_y_preprocesar(
                datasets_por_año[year], 
                year
            )
        
        # 7. Conectar a PostgreSQL
        engine = conectar_postgres()
        
        # 8. Subir a PostgreSQL
        subir_a_postgres(datasets_por_año, engine)
        
        print("\n" + "="*60)
        print("PROCESO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\nTablas creadas en PostgreSQL:")
        for year in TARGET_YEARS:
            print(f"  - paises_{year}")
        
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR EN EL PROCESO")
        print("="*60)
        print(f"\n{str(e)}")
        raise


if __name__ == "__main__":
    main()
