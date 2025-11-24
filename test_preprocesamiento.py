"""
Script de prueba para verificar el preprocesamiento de datos sin conexión a PostgreSQL.
Este script te permite probar todo el pipeline de preprocesamiento localmente.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Importar funciones del script principal
from cargar_postgres import (
    cargar_datos_raw,
    reestructurar_datos,
    filtrar_por_años,
    filtrar_por_indicadores,
    pivotar_por_indicador,
    limpiar_y_preprocesar,
    TARGET_YEARS
)


def main():
    """
    Ejecuta el pipeline de preprocesamiento sin subir a PostgreSQL.
    """
    print("\n" + "="*60)
    print("PRUEBA DE PREPROCESAMIENTO LOCAL (SIN POSTGRES)")
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
        datasets_procesados = {}
        for year in TARGET_YEARS:
            datasets_procesados[year] = limpiar_y_preprocesar(
                datasets_por_año[year], 
                year
            )
        
        # 7. Guardar resultados localmente en CSV para inspección
        print("\n" + "="*60)
        print("GUARDANDO RESULTADOS LOCALMENTE")
        print("="*60)
        
        for year, df in datasets_procesados.items():
            filename = f"paises_{year}_procesado.csv"
            df.to_csv(filename, index=False)
            print(f"\n✓ Guardado: {filename}")
            print(f"  - Países: {len(df)}")
            print(f"  - Columnas: {len(df.columns)}")
            
            # Mostrar primeras filas
            print(f"\nPrimeras 3 filas del año {year}:")
            print(df[['Country Name', 'gdp_per_capita', 'gdp_growth', 'population']].head(3).to_string(index=False))
        
        # 8. Estadísticas comparativas
        print("\n" + "="*60)
        print("ESTADÍSTICAS COMPARATIVAS")
        print("="*60)
        
        for year in TARGET_YEARS:
            df = datasets_procesados[year]
            print(f"\nAño {year}:")
            print(f"  - Países: {len(df)}")
            print(f"  - Variables numéricas: {len(df.select_dtypes(include=[np.number]).columns)}")
            print(f"  - PIB per cápita medio: ${df['gdp_per_capita'].mean():,.2f}")
            print(f"  - PIB per cápita mediano: ${df['gdp_per_capita'].median():,.2f}")
            print(f"  - Población total: {df['population'].sum():,.0f}")
        
        print("\n" + "="*60)
        print("PRUEBA COMPLETADA EXITOSAMENTE")
        print("="*60)
        print("\nLos archivos CSV procesados están listos para inspección.")
        print("Si todo se ve bien, puedes ejecutar cargar_postgres.py para subir a la base de datos.")
        
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR EN LA PRUEBA")
        print("="*60)
        print(f"\n{str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
