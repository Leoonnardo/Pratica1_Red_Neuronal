from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import csv

# Lectura del dataset con la libreria de CSV
def leerDataset():
    tempX = []
    tempY = []
    tempXY = []

    with open('datasetP1.csv', newline='') as File:  
        reader = csv.reader(File)
        auxLit = []
        auxLit1 = []
        listaDatos = []
        listaDatos1 = []
        for row in reader:
            for datos in row:
                auxLit.append(datos)

            # Añadimos las columnas del dataset en una lista individual
            listaDatos.append(auxLit[0])
            listaDatos1.append(auxLit[1])
            auxLit = [] 

    print("listaDatos", listaDatos)
    print("listaDatos1", listaDatos1)

    # Se meten los datos en un array para el manejo de los floats
    tempX = np.array(listaDatos, dtype=float)
    tempY = np.array(listaDatos1, dtype=float)

    # Se juntan los arreglos de X y Y
    tempXY = [tempX, tempY]

    # Y se retorna
    return tempXY

def entrenamiento():
    prediccion = 28
    # Se obtiene el dataset retornado
    dataset = leerDataset()

    print("==== Inicia iteraciones de entrenamiento ====")
    modelo = implementacionKeras()

    # Compilar el algoritmo obteniendo la perdida cuadratica y realiza la optimizacion de que tanto aprendera el algoritmo
    modelo.compile(
        optimizer = keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )

    # Se obtiene el historial del aprendizaje con base a las epocas o numero de iteraciones
    historial = modelo.fit(dataset[0], dataset[1], epochs=1000, verbose= False)

    # Se manda a llamar el interfaz que demostrara la magnitud de perdida y las epocas
    resultados(historial)

    # Demostracion de una prediccion con base a un numero
    print('Prediccion en base a un numero: ', prediccion)
    resultado = modelo.predict([prediccion])
    print('Resultado: ', resultado[0])

def implementacionKeras():

    # Red neuronal Keras
    # Funsion de activacion
    capa = keras.layers.Dense(units= 1, input_shape=[1], activation='selu')
    # Modelo de la grafica a utilizar
    modelo = keras.Sequential([capa])

    return modelo

def resultados(historial):
    plt.xlabel('Numero de epoca')
    plt.ylabel('Magnitud de perdida')
    plt.plot(historial.history['loss'])
 
    plt.show()

entrenamiento()