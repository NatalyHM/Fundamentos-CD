import numpy as np
from scipy import stats
from collections import Counter
import math

# ----- v1: Cálculo Manual -----

def calcular_media_v1(datos):
    return sum(datos) / len(datos)

def calcular_mediana_v1(datos):
    datos_ordenados = sorted(datos)
    n = len(datos)
    mitad = n // 2
    if n % 2 == 0:
        return (datos_ordenados[mitad - 1] + datos_ordenados[mitad]) / 2
    else:
        return datos_ordenados[mitad]

def calcular_moda_v1(datos):
    frecuencia = Counter(datos)
    moda = max(frecuencia, key=frecuencia.get)
    return moda

def calcular_varianza_v1(datos):
    media = calcular_media_v1(datos)
    suma_diferencias_cuadradas = sum((x - media) ** 2 for x in datos)
    return suma_diferencias_cuadradas / (len(datos) - 1)

def calcular_desviacion_estandar_v1(datos):
    varianza = calcular_varianza_v1(datos)
    return math.sqrt(varianza)

def calcular_coeficiente_variacion_v1(datos):
    media = calcular_media_v1(datos)
    desviacion_estandar = calcular_desviacion_estandar_v1(datos)
    return (desviacion_estandar / media) * 100

def normalizacion_z_v1(datos):
    media = calcular_media_v1(datos)
    desviacion_estandar = calcular_desviacion_estandar_v1(datos)
    return [(x - media) / desviacion_estandar for x in datos]


# ----- v2: Cálculo usando Numpy y Scipy -----

def calcular_media_v2(datos):
    return np.mean(datos)

def calcular_mediana_v2(datos):
    return np.median(datos)

# Función corregida para calcular la Moda con Scipy
def calcular_moda_v2(datos):
    moda_resultado = stats.mode(datos)
    if moda_resultado.mode.size == 1:
        return moda_resultado.mode.item()  # Accedemos al valor como escalar
    else:
        return moda_resultado.mode[0]

def calcular_varianza_v2(datos):
    return np.var(datos, ddof=1)

def calcular_desviacion_estandar_v2(datos):
    return np.std(datos, ddof=1)

def calcular_coeficiente_variacion_v2(datos):
    media = calcular_media_v2(datos)
    desviacion_estandar = calcular_desviacion_estandar_v2(datos)
    return (desviacion_estandar / media) * 100

# Ajustamos la función de normalización Z para evitar el tipo np.float64 en la salida
def normalizacion_z_v2(datos):
    media = calcular_media_v2(datos)
    desviacion_estandar = calcular_desviacion_estandar_v2(datos)
    # Convertimos el array a lista de floats normales
    return ((np.array(datos) - media) / desviacion_estandar).tolist()


# ----- Ejemplo de uso -----
datos = [10, 20, 20, 30, 30, 30, 40, 50, 50, 60]

# Resultados usando v1 (Cálculo Manual)
print("---- Resultados con v1 (Cálculo Manual) ----")
print("Media v1:", calcular_media_v1(datos))
print("Mediana v1:", calcular_mediana_v1(datos))
print("Moda v1:", calcular_moda_v1(datos))
print("Varianza v1:", calcular_varianza_v1(datos))
print("Desviación Estándar v1:", calcular_desviacion_estandar_v1(datos))
print("Coeficiente de Variación v1:", calcular_coeficiente_variacion_v1(datos), "%")
print("Normalización Z v1:", normalizacion_z_v1(datos))

# Resultados usando v2 (Numpy y Scipy)
print("\n---- Resultados con v2 (Numpy y Scipy) ----")
print("Media v2:", calcular_media_v2(datos))
print("Mediana v2:", calcular_mediana_v2(datos))
print("Moda v2:", calcular_moda_v2(datos))
print("Varianza v2:", calcular_varianza_v2(datos))
print("Desviación Estándar v2:", calcular_desviacion_estandar_v2(datos))
print("Coeficiente de Variación v2:", calcular_coeficiente_variacion_v2(datos), "%")
print("Normalización Z v2:", normalizacion_z_v2(datos))
