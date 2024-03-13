import matplotlib.pyplot as plt
from  pandas import DataFrame
def plotCurveROC(fpr, tpr):
    """# Autor: Mauricio Gomez S.
    # Fecha: 12-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo generar una grafica de la curva ROC
    #
    # Parametros: 
    #             fpr [array] dataframe para realizar un countplot.
    #             tpr [array] valor con el que se quieren comparar datos.
    #
    # Return: 
    #            [fig] grafica Curva ROC.
    #"""
    plt.plot(false_positive, true_positive, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plt.ylabel('Verdadero Positivo')
    plt.xlabel('Falso Positivo')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()

#
def calcSignificantVars(model, pvalue = 0.05):
    """# Autor: Mauricio Gomez S.
    # Fecha: 25-07-2022
    # Descripcion: La siguiente funcion tiene por obtener las variables significativas
    #              de un modelo de acuerdo a un pvalue propuesto (por defecto se utilizara 0.05)
    #
    # Parametros: 
    #             model [OLSResults] modelo a analizar.
    #             pvalue [float] valor de pvalue a utilizar.
    #
    # Return: 
    #            [dataframe] dataframe de resultado de filtrado por valor de significancia.
    #"""

    df = DataFrame(model.summary2().tables[1]).round(3)
    df = df[df['P>|t|'] < pvalue].sort_values('P>|t|', ascending = True)
    df = df[~df.index.isin(['Intercept'])]
    return df

#

#

#

#
