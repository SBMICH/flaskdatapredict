import pandas as pd

def convert_to_arff(input_file, output_file):
    # Leer el archivo Excel
    data = pd.read_excel(input_file)

    # Abrir el archivo ARFF para escribir
    with open(output_file, 'w') as arff_file:
        # Escribir el encabezado del ARFF
        arff_file.write('@relation FGR_dataset\n\n')
        for column in data.columns:
            arff_file.write(f'@attribute {column} numeric\n')
        arff_file.write('\n@data\n')

        # Escribir los datos
        for _, row in data.iterrows():
            arff_file.write(','.join(map(str, row.values)) + '\n')

    print(f"Archivo ARFF guardado como: {output_file}")

# Especifica los archivos de entrada y salida
input_file = 'FGR_dataset.xlsx'
output_file = 'FGR_dataset.arff'

convert_to_arff(input_file, output_file)