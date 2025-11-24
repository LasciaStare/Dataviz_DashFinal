"""
Módulo de consultas a la base de datos PostgreSQL.

Este módulo proporciona funciones para leer datos de las tablas en PostgreSQL
y devolverlos como DataFrames de pandas para su uso en el dashboard.

Principio de diseño: "Cargar una vez, filtrar en memoria"
- Las funciones de este módulo se llaman UNA SOLA VEZ al iniciar la aplicación
- Los datos se cargan completamente en memoria como DataFrames
- Toda la interactividad del dashboard opera sobre estos DataFrames en memoria
- No se realizan consultas adicionales durante la ejecución del dashboard

Autor: Proyecto Dashboard Arquetipos Economía Global
Fecha: Noviembre 2025
"""

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import sys

# Cargar variables de entorno
load_dotenv()


def get_engine():
    """
    Crea y devuelve un motor de conexión a PostgreSQL.
    
    IMPORTANTE: Esta función utiliza la variable de entorno DB_URL (o DATABASE_URL)
    que debe estar definida en el archivo .env (local) o en las variables de 
    entorno de Render (producción).
    
    Returns:
        sqlalchemy.engine.Engine: Motor de conexión a la base de datos
        
    Raises:
        ValueError: Si faltan credenciales en el archivo .env
        Exception: Si hay error al conectar a la base de datos
    """
    # Obtener URL de conexión desde variables de entorno
    # Soporta tanto DB_URL como DATABASE_URL (común en Render)
    db_url = os.getenv('DB_URL') or os.getenv('DATABASE_URL')
    
    # Validar que la URL esté presente
    if not db_url:
        raise ValueError(
            "Falta la variable de entorno DB_URL o DATABASE_URL\n"
            "Asegúrate de definir en .env: DB_URL=postgresql://user:password@host:port/database"
        )
    
    try:
        # Crear engine con configuración optimizada
        engine = create_engine(
            db_url,
            pool_pre_ping=True,  # Verifica conexiones antes de usarlas
            pool_recycle=3600,   # Recicla conexiones cada hora
            echo=False           # No mostrar SQL queries en producción
        )
        return engine
    except Exception as e:
        raise Exception(f"Error al crear el motor de conexión: {e}")


def cargar_datos_iniciales():
    """
    Carga TODOS los datos necesarios desde PostgreSQL en memoria.
    
    Esta función debe ser llamada UNA SOLA VEZ al inicio de la aplicación Dash.
    Retorna un diccionario con los DataFrames de ambos años, listos para ser
    almacenados en memoria y utilizados por todos los callbacks del dashboard.
    
    Returns:
        dict: Diccionario con estructura:
              {
                  'df_2007': DataFrame con datos del año 2007,
                  'df_2022': DataFrame con datos del año 2022,
                  'metadata': {
                      'num_paises_2007': int,
                      'num_paises_2022': int,
                      'variables': list,
                      'fecha_carga': str
                  }
              }
    
    Raises:
        Exception: Si hay error al cargar los datos
        
    Example:
        >>> datos = cargar_datos_iniciales()
        >>> df_2007 = datos['df_2007']
        >>> df_2022 = datos['df_2022']
    """
    print("\n" + "="*60)
    print("CARGANDO DATOS DESDE POSTGRESQL")
    print("="*60)
    
    try:
        # Obtener engine
        engine = get_engine()
        
        # Verificar conexión
        print("\n1. Verificando conexión a la base de datos...")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("   ✓ Conexión exitosa")
        
        # Cargar datos de 2007
        print("\n2. Cargando datos del año 2007...")
        with engine.connect() as conn:
            df_2007 = pd.read_sql("SELECT * FROM paises_2007", conn)
            print(f"   ✓ {len(df_2007)} países cargados para 2007")
        
        # Cargar datos de 2022
        print("\n3. Cargando datos del año 2022...")
        with engine.connect() as conn:
            df_2022 = pd.read_sql("SELECT * FROM paises_2022", conn)
            print(f"   ✓ {len(df_2022)} países cargados para 2022")
        
        # Validar que los datos no estén vacíos
        if df_2007.empty or df_2022.empty:
            raise Exception("Una o más tablas están vacías. Verifica la base de datos.")
        
        # Extraer lista de variables (excluir columnas de identificación)
        variables = [col for col in df_2007.columns 
                    if col not in ['Country Name', 'Country Code', 'year']]
        
        # Preparar metadata
        from datetime import datetime
        metadata = {
            'num_paises_2007': len(df_2007),
            'num_paises_2022': len(df_2022),
            'variables': variables,
            'fecha_carga': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print("\n4. Resumen de datos cargados:")
        print(f"   - Año 2007: {metadata['num_paises_2007']} países")
        print(f"   - Año 2022: {metadata['num_paises_2022']} países")
        print(f"   - Variables económicas: {len(variables)}")
        print(f"   - Fecha de carga: {metadata['fecha_carga']}")
        
        print("\n" + "="*60)
        print("DATOS CARGADOS EXITOSAMENTE EN MEMORIA")
        print("="*60 + "\n")
        
        # Retornar diccionario con todos los datos
        return {
            'df_2007': df_2007,
            'df_2022': df_2022,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"\n✗ ERROR al cargar datos: {e}")
        print("="*60 + "\n")
        raise


def verificar_conexion():
    """
    Verifica la conexión a la base de datos.
    
    Útil para diagnosticar problemas de conexión antes de intentar cargar datos.
    
    Returns:
        bool: True si la conexión es exitosa, False en caso contrario
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✓ Conexión a PostgreSQL exitosa")
            return True
    except Exception as e:
        print(f"✗ Error de conexión a PostgreSQL: {e}")
        return False


def listar_tablas():
    """
    Lista todas las tablas disponibles en la base de datos.
    
    Útil para verificar que las tablas paises_2007 y paises_2022 existen.
    
    Returns:
        list: Lista con los nombres de las tablas
    """
    try:
        engine = get_engine()
        
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            tables = [row[0] for row in result]
        
        print(f"Tablas encontradas: {tables}")
        return tables
        
    except Exception as e:
        print(f"✗ Error al listar tablas: {e}")
        raise


def obtener_info_tabla(table_name):
    """
    Obtiene información detallada sobre una tabla específica.
    
    Args:
        table_name (str): Nombre de la tabla
        
    Returns:
        dict: Información de la tabla (columnas, tipos, etc.)
    """
    try:
        engine = get_engine()
        
        query = f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        
        with engine.connect() as conn:
            df_info = pd.read_sql(query, conn)
        
        return df_info.to_dict('records')
        
    except Exception as e:
        print(f"✗ Error al obtener info de la tabla {table_name}: {e}")
        raise


# Función de prueba para verificar que todo funciona
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "VERIFICACIÓN DEL MÓDULO DE CONSULTAS")
    print("="*70)
    
    # Verificar conexión
    print("\n1. Verificando conexión a PostgreSQL...")
    if not verificar_conexion():
        print("\n✗ No se pudo conectar a la base de datos.")
        print("Verifica que:")
        print("  - El archivo .env existe y contiene DB_URL")
        print("  - La base de datos en Render está activa")
        print("  - Las credenciales son correctas")
        sys.exit(1)
    
    # Listar tablas
    print("\n2. Listando tablas disponibles...")
    try:
        tables = listar_tablas()
        if 'paises_2007' not in tables or 'paises_2022' not in tables:
            print("\n⚠ ADVERTENCIA: No se encontraron las tablas esperadas")
            print(f"Tablas encontradas: {tables}")
            print("Ejecuta primero: python cargar_postgres.py")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    # Cargar datos iniciales (simulando el inicio de la aplicación)
    print("\n3. Ejecutando carga inicial de datos (como en dashboard.py)...")
    try:
        datos = cargar_datos_iniciales()
        
        # Verificar contenido
        print("\n4. Verificando contenido de los DataFrames...")
        print(f"\n   DataFrame 2007:")
        print(f"   - Shape: {datos['df_2007'].shape}")
        print(f"   - Columnas: {list(datos['df_2007'].columns)}")
        print(f"   - Primeros países: {datos['df_2007']['Country Name'].head(3).tolist()}")
        
        print(f"\n   DataFrame 2022:")
        print(f"   - Shape: {datos['df_2022'].shape}")
        print(f"   - Columnas: {list(datos['df_2022'].columns)}")
        print(f"   - Primeros países: {datos['df_2022']['Country Name'].head(3).tolist()}")
        
        print("\n5. Metadata:")
        for key, value in datos['metadata'].items():
            if key != 'variables':
                print(f"   - {key}: {value}")
            else:
                print(f"   - {key}: {len(value)} variables")
        
        print("\n" + "="*70)
        print(" "*20 + "✓ VERIFICACIÓN COMPLETADA")
        print(" "*15 + "El módulo está listo para usar en dashboard.py")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error durante la carga: {e}")
        sys.exit(1)

