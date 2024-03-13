#!/usr/bin/env python
# coding: utf-8

import skimpy
from skimpy import clean_columns
from pandas import DataFrame
from numpy import nan

def convert_float(string):
    """# Autor: Mauricio Gomez S.
    # Fecha: 30-11-2022
    # Descripcion: La siguiente funcion tiene por objetivo eliminar . (punto) de atributos
    #              de tipo objeto de la forma 91.800.861 para transformarlos a 91.800861
    #              de tipo float
    #
    # Parametros: 
    #             string [str] atributo de entrada al que se le eliminaran los . (puntos)
    #
    # Return: 
    #            numero [float] atributo normalizado, en caso contrario devolvera valor perdido
    #"""
    try:
        numero = float(string[0:string.find('.')] + '.' + string[string.find('.'):].replace('.', ''))
    except:
        numero = nan
    return numero

def dropUnnamedCols(df):
    """# Autor: Mauricio Gomez S.
    # Fecha: 13-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo quitar las columnas que comienzan
    #              con Unnamed
    #
    # Parametros: 
    #             df [dataframe] dataframe con columnas que comiencen por Unnamed.
    #
    # Return: 
    #            df [dataframe] dataframe sin columnas que comiencen por Unnamed
    #"""
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
def is_empty(data_structure):
    """# Autor: Mauricio Gomez S.
    # Fecha: 13-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo validar si la estructura de datos
    #              preguntada contiene datos o esta vacia
    #
    # Parametros: 
    #             data_structure [list, dict, tuple] estructura de datos a validar.
    #
    # Return: 
    #            True: si la estructura de datos esta vacia
    #            False: si la estructura de datos no esta vacia.
    #"""
    if data_structure:
        return False
    else:
        return True

def dropNaN(df, dropType):
    """# Autor: Mauricio Gomez S.
    # Fecha: 13-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo eliminar valores perdidos del
    #              dataframe
    #
    # Parametros: 
    #             df [dataframe] dataframe a normalizar.
    #             dropType [str] es el tipo de eliminacion que se desea aplicar.
    #                   - row: se utiliza para eliminar filas que contengan al menos 1
    #                          valor perdido
    #                   - col: se utiliza para eliminar columnas que contengan al menos 1
    #                          valor perdido
    #
    # Return: 
    #            df [dataframe] dataframe sin valores perdidos
    #"""
    if(dropType == 'row'):
        df = df.dropna(axis = 0)
    if(dropType == 'col'):
        df = df.dropna(axis = 1)
    return df

def whitespaceRemover(dataframe):
    """# Autor: Mauricio Gomez S.
    # Fecha: 01-08-2022
    # Descripcion: La siguiente funcion tiene por objetivo eliminar los espacios tanto a la derecha 
    #              como a la izquierda de un dataframe
    #
    # Parametros: 
    #             df [dataframe] dataframe a normalizar.
    #
    # Return: 
    #            df [dataframe] dataframe sin espacios
    #"""
    # iterating over the columns
    for i in dataframe.columns:
        # checking datatype of each columns
        if dataframe[i].dtype == 'object':
            # applying strip function on column
            dataframe[i] = dataframe[i].map(str.strip)
        else:
            # if condn. is False then it will do nothing.
            pass

#

#

#

#

#
