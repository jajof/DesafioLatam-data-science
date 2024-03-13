import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score



# Función 1
def estandarizar_datos_nulos(df, lista_nulos, cols_comillas = None):
    """ Esta función estandariza los nulos para un dataframe, además 
        de eliminar comillas para datos numéricos mal ingresados.

    Parameters
    ----------
    df [DataFrame]:
        Set de datos a limpiar. 

    lista_nulos [List]:
        Lista de strings que deben ser reemplazados por np.nan
    
    cols_comillas = None [List]:
        Lista de columnas con comillas a ser eliminadas
    
    Returns
    -------
    [DataFrame]
        Retorna el mismo DataFrame, pero corregido
    """
    lista_corregida = [np.nan for _ in lista_nulos]
    df = df.replace(lista_nulos, lista_corregida)
    if cols_comillas:
        for col in cols_comillas:
            df[col] = df[col].str.replace('"', '')
    return df


# Función 2
def limpiar_dataFrame(df, lista_columnas_relevantes):
    """ Elimina los nulos para las columnas relevantes. Además, intenta
    traspasar los valores a int. Notifica qué columnas pudo traspasar

    Parameters
    ----------
    df [DataFrame]:
        Set de datos a trabajar. 

    lista_columnas_relevantes [List]:
        Lista de columnas relevantes y que serán incluidas en
        el DataFrame final

    Returns
    -------
    [DataFrame]
        Retorna el mismo DataFrame solo con las columnas relevantes sin Nan
    """
    df = df.copy()[lista_columnas_relevantes]
    df = df.dropna()
    for c in df:
        try:
            df[c] = df[c].astype(int)
            print(f'Columna {c} transformada a entero')
        except:
            print(f'Columna {c} no pudo ser transformada')

    df = df.reset_index(drop = True)
    return df

# Función 3
def inverse_logit(x):
    """ Aplica la definición de inversa de logit para un X

    Parameters
    ----------
    x [float]:
        valor a utilizar en la función. Debiese ser un logodss

    Returns
    -------
    [float]
        Retorna la probabilidad asociada al logodds
    """    
    return 1 / (1 + np.exp(-x))


# Función 4
def generar_formula_regresion(lista_independientes, dependiente):
    """ Genera la fórmula para la regresión de smf.logit o smf.ols

    Parameters
    ----------
    lista_independientes [List]:
        Lista de columnas que debe ser utilizada como regresores
    
    dependiente [string]:
        Nombre de la variable dependiente

    Returns
    -------
    [string]
        Retorna el string que puede ser utilizado en smf.logit o smf.ols
    """    
    aux = ''
    for i, columna in enumerate(lista_independientes):
        if columna != dependiente:
            if i == 0:
                aux = columna
            else:
                aux = aux + f' + {columna}'
    return dependiente + ' ~ ' + aux


def calcular_log_odds(modelo, diccionario_valores):
    """ Calcula el log odds para cierto modelo y un individuo definido por un diccionario 
    con los valores para cada atributo

    Parameters
    ----------
    Modelo [statsmodels.discrete.discrete_model.BinaryResultsWrapper]:
        Modelo de stats models
    
    diccionario_valores [string]:
        Diccionario que caracteriza al individuo cuyo log ods se quiere calcular

    Returns
    -------
    [float]
        Retorna el log odds asociado al individuo para el modelo
    """    
    acumulado = 0
    for nombre, regresor in zip(modelo.params.index, modelo.params.values):
        if nombre == 'Intercept':
            acumulado = acumulado + regresor
        else:
            acumulado = acumulado + regresor * diccionario_valores[nombre]
    return acumulado

def report_scores(y_hat, y_test):
    """ Imprime el error cuadrático promedio, su raiz cuadrada y el R cuadrado

    Parameters
    ----------
    y_hat [np.array]:
        Vector de predicciones
    
    y_test [np.array]:
        Vector de valores reales del conjunto de Test

    Returns
    -------
    None
    """   
    print("Error Cuadrático Promedio: {0}".format(mean_squared_error(y_test, y_hat, squared=True).round(1)))
    print(f"Raiz del Error Cuadrático Promedio: {np.sqrt(mean_squared_error(y_test, y_hat, squared=True)).round(1)}")
    print("R2: {0}".format( r2_score(y_test, y_hat).round(1)))








if __name__ == '__main__':
    df = pd.read_csv('students.csv', sep = '|').drop(columns = ['Unnamed: 0'])
    df = estandarizar_datos_nulos(df, ['nulidade', 'sem validade', 'zero'], ['age', 'goout', 'health'])
    print(df['age'].value_counts())
