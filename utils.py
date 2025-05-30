import pandas as pd
# === Improved product grouping with better balancing ===
def agrupar_productos(df, num_grupos=8):
    """Group products with weighted balancing of quantity and volume"""
    grupos = [[] for _ in range(num_grupos)]
    suma_cantidades = [0] * num_grupos
    suma_volumen = [0] * num_grupos
    conteo_productos = [0] * num_grupos
    
    # Sort by multiple criteria for better balancing
    df = df.sort_values(['UNDESTIMADAS', 'VOLUMEN'], ascending=[False, False]).reset_index(drop=True)
    
    for _, row in df.iterrows():
        # Use a weighted score of quantity and volume for better balancing
        mejor_grupo = min(range(num_grupos), 
                         key=lambda g: (suma_cantidades[g]/max(1, sum(suma_cantidades))*0.5 + 
                                       suma_volumen[g]/max(1, sum(suma_volumen))*0.3 +
                                       conteo_productos[g]/max(1, sum(conteo_productos))*0.2))
        grupos[mejor_grupo].append(row)
        suma_cantidades[mejor_grupo] += row['UNDESTIMADAS']
        suma_volumen[mejor_grupo] += row['VOLUMEN']
        conteo_productos[mejor_grupo] += 1
    
    # Print group statistics for verification
    for i, grupo in enumerate(grupos):
        group_df = pd.DataFrame(grupo)
        print(f"Group {i+1}: {len(grupo)} products, " +
              f"Total quantity: {suma_cantidades[i]:.2f}, " +
              f"Total volume: {suma_volumen[i]:.2f}")
    
    return [pd.DataFrame(g) for g in grupos]


def get_data():
    """Load and preprocess product data with enhanced normalization"""
    file_path = 'productos_anaquel.xlsx'
    df_list = []
    i = 1
    try:
        while True:
            df_list.append(pd.read_excel(file_path, sheet_name=f"Sheet {i}"))
            i += 1
    except Exception as e:
        pass

    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all[df_all['ANAQUEL'].str.startswith('C', na=False)]
    #print(df_all.head())
    # Data cleaning and normalization
    df_all['UNDESTIMADAS'] = df_all['UNDESTIMADAS'].apply(lambda x: max(x, 1))  # Ensure positive
    df_all = df_all.drop_duplicates(subset='PRODUCTO')
    
    exclude_campa = [201915, 201916, 201917, 201918, 202002, 202003, 202005, 202006,
                     202008, 202009, 202011, 202010, 202012, 202004, 202007]
    df_all = df_all[~df_all['CAMPA'].isin(exclude_campa)]

    df_all = df_all[['PRODUCTO','ALTO', 'ANCHO', 'LARGO', 'VOLUMEN', 'PESO', "UNDESTIMADAS", 'CAMPA', 'ANAQUEL']]
    
    
    # Robust normalization: use min-max but handle outliers
    for col in ['ALTO', 'ANCHO', 'LARGO', 'VOLUMEN', 'PESO', 'UNDESTIMADAS']:
        # Calculate percentiles to handle outliers
        q_low = df_all[col].quantile(0.1)
        q_high = df_all[col].quantile(0.8)
        
        # Clip values to reduce impact of outliers
        df_all[col] = df_all[col].clip(q_low, q_high)
        
        # Apply min-max normalization
        df_all[col] = (df_all[col] - df_all[col].min()) / (df_all[col].max() - df_all[col].min() + 1e-8)
    
    # print(df_all.describe())
    df_all.reset_index(drop=True, inplace=True)
    return df_all