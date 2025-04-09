import pandas as pd

# Especifica el nombre del archivo original y el nombre del nuevo archivo
input_file = 'productos_anaquel.xls'
output_file = 'productos_anaquel.xlsx'

# Lee todas las hojas del archivo original
xls = pd.ExcelFile(input_file)

# Crea un writer para el nuevo archivo Excel
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Conversión completa: '{input_file}' -> '{output_file}'")