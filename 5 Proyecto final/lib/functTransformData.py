#!/usr/bin/env python
# coding: utf-8

# manipulacion de datos
from numpy import where
from pandas import cut
# tratamieno de fechas
from datetime import datetime, timedelta
import flatten_json

def ageRange(age):
    """# Autor: Mauricio Gomez S.
    # Fecha: 25-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo categorizar la edad
    #
    # Params: 
    #             model [OLSResults] modelo a analizar.
    #             pvalue [float] valor de pvalue a utilizar.
    #
    # Return: 
    #            [dataframe] dataframe de resultado de filtrado por valor de significancia.
    #"""
    # edad
    lstRangeAge = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    lstCatAge = ["0 - 10", "10 - 20", "20 - 30", "30 - 40", "40 - 50", "50 - 60", "60 - 70", "70 - 80", "80 - 90", "90 - 100"]
    lstRangeAgeType = [0, 2, 11, 17, 64, 100]
    # lstAgeType = ['bebe', 'niño', 'adolescente', 'adulto', 'Anciano']
    lstAgeType = ['baby', 'child', 'adolescent', 'adult', 'elderly']
    lstRangeGeneration = [1930, 1949, 1969, 1980, 1993, 2010, 2022]
    # lstGeneration = ['Niños post guerra', 'Baby boomer', 'Generación X', 'Millennials', 'Generacion Z']
    lstGeneration = ['Post war children', 'Baby boomer', 'Generation X', 'Millennials', 'Generation Z', 'No information']
    if((age < 0) | (age > 100)):
        age = 0                                          # posicion 0
        ageRange = ['No information']                    # posicion 1
        typeAgeRange = ['No information']                # posicion 2
        yearBirth = ['No information']                   # posicion 3
        ageGeneration = ['No information']               # posicion 4
    else:
        ageRange = cut([age], lstRangeAge, labels = lstCatAge)                     # posicion 1
        typeAgeRange = cut([age], lstRangeAgeType, labels = lstAgeType)            # posicion 2
        yearBirth = [(datetime.now() - timedelta(days=(age*365))).year]            # posicion 3
        ageGeneration = cut(yearBirth, lstRangeGeneration, labels = lstGeneration) # posicion 4

    return age, ageRange[0], typeAgeRange[0], yearBirth[0], ageGeneration[0]

def binarizeDf(df):
    """# Autor: Mauricio Gomez S.
    # Fecha: 13-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo binarizar columnas del 
    #              dataframe
    #
    # Params: 
    #             df [dataframe] dataframe a normalizar.
    #
    # Return: 
    #            df [dataframe] dataframe binarizado
    #            print [dataframe] imprime resultado de la binarizacion por variable
    #"""
    for i in df.columns:
        if len(df[i].value_counts('%')) ==2:
            df[i] = df[i].replace([df[i].value_counts('%').index[0],df[i].value_counts('%').index[1]],[0,1])
            dfTemporal = DataFrame({'total': df[i].value_counts(), 'percent': df[i].value_counts('%').round(3)})
            print(f"\033[1mNombre de variable: \033[0m{i}, \033[1mfrecuencia: \033[0m\n{dfTemporal}\n")

def extractNums(celda):
    """# Autor: Mauricio Gomez S.
    # Fecha: 01-08-2022
    # Descripcion: La siguiente funcion tiene por objetivo extraer numeros de un string 
    #
    # Params: 
    #             celda [string] cadena de texto a analizar.
    #
    # Return: 
    #            resultado [int] resultado de la extraccion de numeros, en caso de
    #            no existir numeros el resultado devolvera un valor 0.
    #"""
    m = re.match(r"\d+", celda)
    if m:
        # Si hubo coincidencia, m.group() devuelve el texto que coincidió
        # Basta convertirlo en entero
        resultado = int(m.group())
        return resultado
    else:
        # Si no hubo coincidencia (lo que ocurre también en celdas vacías)
        # el valor a retornar es cero
        resultado = 0
        return resultado

    
def flatten_json(nested_json: dict, exclude: list=[''], sep: str='_') -> dict:
    """# Autor: amirziai
    # Fecha: 20-02-2021
    # Descripcion: La siguiente funcion tiene por objetivo transformar un JSON a DataFrame
    #
    # Params: 
    #             json [dict] json a transformar.
    #             list [lst] lista de atributos que se quieren excluir.
    #             sep [str] separador.
    #
    # Return: 
    #            out [dict] diccionario producto de la transformacion
    #"""
    out = dict()
    def flatten(x: (list, dict, str), name: str='', exclude=exclude):
        if type(x) is dict:
            for a in x:
                if a not in exclude:
                    flatten(x[a], f'{name}{a}{sep}')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, f'{name}{i}{sep}')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out