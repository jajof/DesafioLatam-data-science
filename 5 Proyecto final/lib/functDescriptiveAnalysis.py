import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
import seaborn as sns
from itertools import combinations


def PlotHistDA(df):
    """# Autor: Mauricio Gomez S.
    # Fecha: 12-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo generar histogramas
    #              de distintas columnas de un dataframe, incluyendo lineas verticales
    #              para representar la media y mediana
    #
    # Parametros: 
    #             df [dataframe] dataframe a representar como histograma.
    #
    # Return: 
    #            [fig] sub plot de los histogramas del dataframe.
    #"""
    colVars = df.columns.to_list()
    listColsRows = list(range(0, len(colVars)))
    listRows = [ x//2 for x in listColsRows]
    listCols = [ x%2 for x in listColsRows]
    totalrows = round(len(colVars)/2)
    if(len(catList) % 2 != 0):
        totalrows = totalrows + 1
    fig, axs = plt.subplots(ncols = 2
                            , nrows = totalrows
                            , figsize=(15, 20)
                            , constrained_layout=True)
    for variable, row, col in zip(colVars, listRows, listCols):
        axs[row, col].hist(df[variable],bins = 25)
        axs[row, col].axvline(round(np.mean(df[variable]), 3),label='Media de la variable: '+ str(round(np.mean(df[variable]), 3)),color='blue')
        axs[row, col].axvline(round(np.median(df[variable]), 3),label='Mediana de la variable: '+ str(round(np.median(df[variable]), 3)),color='r')
        axs[row, col].legend()
        axs[row, col].set_title("Histograma de: " + variable)
    fig.suptitle('plt.subplots()')

    
def inspectDataFrameDA(df):
    """# Autor: Mauricio Gomez S.
    # Fecha: 12-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo generar un analisis descriptivo
    #              de distintas columnas de un dataframe, separando el analisis entre variables
    #              continuas y discretas
    #
    # Parametros: 
    #             df [dataframe] dataframe a analizar de forma descriptiva.
    #
    # Return: 
    #            [print] para las variables float64 e int64 entrega la siguiente informacion
    #                    en una tabla impresa:
    #                        count, mean, std, min, 25%, 50%, 75%, max
    #                    Para las variables distintas a float64 e int64 entrega la siguiente 
    #                    informacion en una tabla impresa:
    #                        valor (unico como indice), total y porcentaje
    #"""
    continuousVars = df.select_dtypes(include=['float64', 'int64'])
    print("\033[1mMedidas descriptivas de variables float64 e int64\033[0m\n")
    try:
        print(round(continuousVars.describe().T, 3))
    except:
        pass
    discreteVars = df.select_dtypes(exclude=['float64', 'int64'])
    print("\n\033[1mFreecuencia de variables distintas a float64 e int64\033[0m\n")
    for i in discreteVars.columns:
        dfTemporal = DataFrame({'total': df[i].value_counts(), 'percent': df[i].value_counts('%').round(3)})
        print(f"\033[1mNombre de variable: \033[0m{i}, \033[1mfrecuencia: \033[0m\n{dfTemporal}\n")

def countNaNDA(dfTemp):
    """# Autor: Mauricio Gomez S.
    # Fecha: 12-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo generar un cuadro resumen del total de
    #              valores perdidos y su respectivo porcentaje
    #
    # Parametros: 
    #             df [dataframe] dataframe a analizar.
    #
    # Return: 
    #            [dataframe] con resultado del analisis de valores perdidos
    #"""
    count_missing = round(dfTemp.isnull().sum(), 0)
    percent_missing = round(dfTemp.isnull().sum() * 100 / len(dfTemp), 3)
    missing_value_df = DataFrame({'total_missing': count_missing, 'percent_missing': percent_missing})
    return missing_value_df.T

def PlotCorrDA(df, colName):
    """# Autor: Mauricio Gomez S.
    # Fecha: 12-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo generar un grafico de correlacion
    #              de distintas columnas de un dataframe comparando una columna en particular
    #
    # Parametros: 
    #             df [dataframe] dataframe a representar como histograma.
    #             colName [str] nombre de la columna que se utilizara para el analisis de correlacion
    #
    # Return: 
    #            [fig] sub plot de la correlacion del dataframe.
    #"""
    corr = df.set_index(colName).corr()
    sm.graphics.plot_corr(corr, xnames=list(corr.columns))
    plt.show()
    
def contentAnalysisDA(df, types = 'discreta', cols = []):
    """# Autor: Mauricio Gomez S.
    # Fecha: 1407-2022
    # Descripcion: La siguiente funcion tiene por objetivo generar un cuadro resumen del % de
    #              los valores contenidos en cada columna discreta ( o no continua
    #
    # Parametros: 
    #             df [dataframe] dataframe a analizar.
    #             types [str] tipo de variable a analizar.
    #             cols [list] lista de nombre de las columnas a analizar.
    #
    # Return: 
    #            [print] para las variables continuas y discretas entrega el % de su contenido
    #"""
    if not cols:
        if(types == 'discreta'):
            for i,j in df.select_dtypes(exclude=['float64', 'int64']).iteritems():
                print('\ncolumna:',i+'\n'+str(j.value_counts('%')))
        if(types == 'continua'):
            for i,j in df.select_dtypes(include=['float64', 'int64']).iteritems():
                print('\ncolumna:',i+'\n'+str(j.value_counts('%')))
    else:
        if(types == 'discreta'):
            for i,j in df[cols].iteritems():
                print('\ncolumna:',i+'\n'+str(j.value_counts('%')))

                
def countPlotDA(df, xValue, catList):
    """# Autor: Mauricio Gomez S.
    # Fecha: 12-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo generar histogramas
    #              de distintas columnas de un dataframe, incluyendo lineas verticales
    #              para representar la media y mediana
    #
    # Parametros: 
    #             df [dataframe] dataframe para realizar un countplot.
    #             xValue [str] valor con el que se quieren comparar datos.
    #             catList [lst] lista de atributos a comparar.
    #
    # Return: 
    #            [fig] sub plot de los countplot del dataframe.
    #"""
    listColsRows = list(range(0, len(catList)))
    listRows = [ x//2 for x in listColsRows]
    listCols = [ x%2 for x in listColsRows]
    totalrows = round(len(catList)/2)
    if(len(catList) % 2 != 0):
        totalrows = totalrows + 1
    fig, axs = plt.subplots(ncols = 2
                            , nrows = totalrows
                            , figsize=(20, 35)
                            , constrained_layout=True)
    for variable, row, col in zip(catList, listRows, listCols):
        sns.countplot(x = xValue
                      , hue = variable
                      , data = df
                      #, order =df[variable].value_counts().index
                      , ax = axs[row, col]
                     )

def countPlotDescriptionDA(df):
    """# Autor: Mauricio Gomez S.
    # Fecha: 12-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo generar histogramas
    #              de distintas columnas de un dataframe, incluyendo lineas verticales
    #              para representar la media y mediana
    #
    # Parametros: 
    #             df [dataframe] dataframe para realizar un countplot.
    #
    # Return: 
    #            [fig] sub plot de los countplot del dataframe.
    #"""
    catList = df.columns.to_list()
    listColsRows = list(range(0, len(catList)))
    listRows = [ x//2 for x in listColsRows]
    listCols = [ x%2 for x in listColsRows]
    totalrows = round(len(catList)/2)
    if(len(catList) % 2 != 0):
        totalrows = totalrows + 1
    fig, axs = plt.subplots(ncols = 2
                            , nrows = totalrows
                            , figsize=(20, 35)
                            , constrained_layout=True)
    for variable, row, col in zip(catList, listRows, listCols):
        sns.countplot(x = variable
                      , data = df
                      , order = df[variable].value_counts().index
                      , ax = axs[row, col]
                     )
        if(df[variable].dtypes != 'object'):
            media = media = np.mean(list(map(int, df[variable].values)))
            axs[row, col].axhline(media, label='Media ' + variable + ': '+ str(media))

        
  





#-------------------------------------------------------------------------------------
# funcion fallida se queda pegada
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
def boxPlotDA(df):
    """# Autor: Mauricio Gomez S.
    # Fecha: 12-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo generar histogramas
    #              de distintas columnas de un dataframe, incluyendo lineas verticales
    #              para representar la media y mediana
    #
    # Parametros: 
    #             df [dataframe] dataframe para realizar un countplot.
    #             xValue [str] valor con el que se quieren comparar datos.
    #             catList [lst] lista de atributos a comparar.
    #
    # Return: 
    #            [fig] sub plot de los countplot del dataframe.
    #"""
    colsListTemp = df.columns.to_list()
    colsList = []
    temp = combinations(colsListTemp, 2)
    for i in list(temp):
        colsList.append(i)
    listColsRows = list(range(0, len(colsList)))
    listRows = [ x//2 for x in listColsRows]
    listCols = [ x%2 for x in listColsRows]
    totalrows = round(len(colsList)/2)
    if(len(colsList) % 2 != 0):
        totalrows = totalrows + 1
    fig, axs = plt.subplots(ncols = 2
                            , nrows = totalrows
                            , figsize=(20, 35)
                            , constrained_layout=True)
    for variable, row, col in zip(colsList, listRows, listCols):
        sns.boxplot(x = variable[0]
                    , y = variable[1]
                    , data = df
                    , ax = axs[row, col]
                   )

