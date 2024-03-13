from sklearn.metrics import mean_squared_error, median_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist(dataframe, var):
    """
    Crea un histograma de la variable var en el dataframe.
    Muestra la media y la mediana en el histograma.
    
    Args:
        dataframe: un dataframe de Pandas
        var: una variable del dataframe
        
    Returns:
        Un histograma de la variable var en el dataframe.
    """
    tmp = dataframe[var].copy()
    sns.histplot(tmp, color='grey', alpha=.2)
    plt.title(var)
    plt.axvline(np.mean(tmp), color='dodgerblue', label ="Media")
    plt.axvline(np.median(tmp), color='tomato', label = "Mediana")
    plt.gca().set(title= var, ylabel= var)
    plt.legend()



def plot_grouped_bar(df, eje, hue, var, figsize):
    """
    Genera un conjunto de gráficos de barras que muestran la cantidad de observaciones y la media de una variable en un conjunto de datos.
    
    Parámetros:
        df (DataFrame): Un dataframe con los datos a representar.
        eje (str): El nombre de la columna que se utilizará en el eje de las abscisas.
        hue (str): El nombre de la columna que se utilizará para agrupar los gráficos por categoría.
        var (str): El nombre de la columna que se representará en el eje de las ordenadas.
        figsize (tuple): Una tupla con el ancho y alto deseado para la figura.
    
    Returns:
        None: La función no retorna ningún valor.
    """
    plt.figure(figsize = figsize)
    df_aux = df.groupby([eje, hue])[var].agg([np.ma.count, np.mean]).reset_index()
    for i, g in enumerate(np.sort(df_aux[hue].unique())):
        tmp = df_aux.copy()[df_aux[hue] == g]
        plt.subplot(len(df_aux[hue].unique()), 1, i + 1)    
        plt.bar(x = tmp[eje], height = tmp['mean'], label = f'Media de {var}')
        plt.plot(tmp[eje], tmp['count'], 'bo', markersize = 12, c = 'orange', label = 'Cantidad observaciones')
        if i == 0:
            plt.legend(prop={'size': 20})
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel(eje, fontsize = 16)
        plt.title(f'Pedidos para {hue} {g}', fontsize = 18)
    plt.tight_layout()



def graficar_error(df_error_aux, figsize, dow = None ):
    """
        Genera un conjunto de gráficos de barras que muestran el error absoluto mediano y la raíz del error cuadrático medio en un conjunto de datos.
        
        Parámetros:
            df_error_aux (DataFrame): Un dataframe con los datos de error a representar.
            figsize (tuple): Una tupla con el ancho y alto deseado para la figura.
            dow (str, optional): Una categoría de día de la semana para filtrar los datos. Si se omite, se representan todos los datos.
        
        Returns:
            None: La función no retorna ningún valor.
    """    
    if dow is None:
        tmp = df_error_aux.copy()
    else:
        tmp = df_error_aux[df_error_aux['dow'] == dow].copy()
        
    df_error_grouped = tmp.groupby(['hora_de_pedido_aprox']).agg(
            {
                'Error': np.median, 
                'Error2': np.mean
            }
        ).reset_index()
    
    df_error_grouped.columns = ['hora_de_pedido_aprox', 'MedianAbsoluteError', 'MeanSquaredError']
    
    df_error_grouped['RaizMeanSquaredError'] = np.sqrt(df_error_grouped['MeanSquaredError'])
    
    plt.figure(figsize = (12, 4))
    plt.subplot(1, 2, 1)
    sns.barplot(
        df_error_grouped,
        x = 'hora_de_pedido_aprox',
        y = 'RaizMeanSquaredError',
        color = 'Orange'
    );
    plt.title(f"Raiz error cuadrático medio {f'para dow {dow}' if dow is not None else ''}");

    plt.subplot(1, 2, 2)
    sns.barplot(
        df_error_grouped,
        x = 'hora_de_pedido_aprox',
        y = 'MedianAbsoluteError',
        color = 'blue'
    );
    plt.title(f"Error absoluto mediano {f'para dow {dow}' if dow is not None else ''}");
    plt.tight_layout();



  
      
    
def plot_importance(fit_model, feat_names, n=10):
    """
        Genera un gráfico con los atributos más importantes de un modelo
        @params:
            fit_model: Requerido. Modelo entrenado.
            feat_names: Requerido. Lista con las columnas usadas en el entrenamiento
            n: Opcional, top de columnas que se desean extraer
        @return:
            list: Retorna una lista con las columnas más importantes
    """
    
    if hasattr(fit_model, 'feature_importances_'):
        tmp_importance = fit_model.feature_importances_
        titulo = 'Feature importance'
    elif hasattr(fit_model, 'coef_'):
        tmp_importance = fit_model.coef_
        titulo = 'Coeficientes'
    else:
        return 'Solo soporta modelos que tienen atributos coef_ o feature_importances_'
    sort_importance = np.argsort(tmp_importance)[::-1][:n] # Retorna un array con n índices más importantes
    names = reversed([feat_names[i] for i in sort_importance]) # obtiene los nombres de las columnas en las posiciones
    plt.title(titulo)
    plt.barh(range(n), list(reversed(tmp_importance[sort_importance]))) # Grafica los valores de las columnas más importantes
    plt.yticks(range(n), names, rotation=0)
    return names