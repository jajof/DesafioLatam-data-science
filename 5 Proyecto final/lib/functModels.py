# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 00:51:17 2022

@author: jahof
"""

from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from lib.functReportGraficas import plot_importance, resultados_modelo



def get_scaled_matrix(X_train, X_test, variables):
    """
    La función get_scaled_matrix es una función que toma tres argumentos: X_train, X_test y variables.
    Los argumentos X_train y X_test son matrices o dataframes de entrenamiento y prueba, respectivamente.
    El argumento variables es una lista de nombres de variables o características que se seleccionarán de X_train y X_test.
    La función hace lo siguiente:
        Crea un objeto scaler de la clase MinMaxScaler y lo ajusta a X_train_dep.
        Al "ajustar" el objeto scaler a X_train_dep, se está calculando el rango mínimo y máximo de los
        datos para poder aplicar la transformación a las matrices de datos.
        Aplica la transformación del escalador a X_train_dep y asigna el resultado a la variable X_train_dep_std.
        Aplica la transformación del escalador a X_test_dep y asigna el resultado a la variable X_test_dep_std.
        Devuelve las matrices transformadas X_train_dep_std y X_test_dep_std.
        La función get_scaled_matrix se utiliza para aplicar una transformación de escalado a las matrices de datos de entrenamiento y prueba.
        Esto puede ser útil en el proceso de preparación de los datos para el entrenamiento de un modelo de aprendizaje automático.
        
     X_train = DataFrame
     X_test = Dataframe
     variables = str
    
    """
    X_train_dep = X_train[variables]
    X_test_dep = X_test[variables]
    
    scaler = MinMaxScaler().fit(X_train_dep)
    X_train_dep_std = scaler.transform(X_train_dep)
    X_test_dep_std = scaler.transform(X_test_dep)
    return X_train_dep_std, X_test_dep_std


def fit_or_load_model(X_train, X_test, y_train, y_test, base_model,
                      grid = {}, fit_or_load = 'fit', con_outliers = True):
    """
    La función fit_or_load_model es una función que se utiliza para entrenar o cargar un modelo de machine learning.
    La función toma varios parámetros de entrada:
    
    X_train: Es una matriz que contiene los datos de entrenamiento para el modelo.
    X_test: Es una matriz que contiene los datos de prueba para el modelo.
    y_train: Es un vector que contiene las etiquetas de entrenamiento para el modelo.
    y_test: Es un vector que contiene las etiquetas de prueba para el modelo.
    base_model: Es el modelo de machine learning que se va a utilizar.
    grid: Es un diccionario que contiene los parámetros que se van a utilizar en la búsqueda de la mejor configuración del modelo a través de validación cruzada.
    fit_or_load: Es una cadena que puede ser 'fit' o 'load'. Si es 'fit', entonces se entrena el modelo y se guarda en disco. Si es 'load', se carga el modelo desde disco.
    con_outliers: Es una bandera que indica si se deben incluir o excluir los valores atípicos en el proceso de entrenamiento del modelo.
    La función realiza lo siguiente:
        Se aplican las transformaciones de escala a los datos de entrenamiento y prueba.
        Si fit_or_load es 'fit', entonces se entrena el modelo con los datos de entrenamiento y se evalúa con los datos de prueba. Se utiliza validación cruzada y se busca la mejor configuración del modelo utilizando el diccionario grid. Luego, se guarda el modelo en disco en la ruta especificada.
        Si fit_or_load es 'load', entonces se cargan los modelos entrenados desde disco y se muestra un gráfico de 
        importancia de variables para el modelo cargado.
        En ambos casos, se entrena un modelo depurado utilizando sólo las variables más relevantes y se guarda en disco.
        
    """

    model_name = str(base_model.__class__).replace("'>", '').split('.')[-1]

    model_name1 = f"{model_name}{'' if con_outliers else '_so'}.sav"
    model_name2 = f"{model_name}_dep{'' if con_outliers else '_so'}.sav"
    
    # Se guardará en la ruta ./Modelos
    path1 = Path("./models")  / model_name1
    path2 = Path("./models")  / model_name2
    
    X_train_std, X_test_std = get_scaled_matrix(X_train, X_test, list(X_test.columns))
    
    if fit_or_load == 'fit':
        print('Comienza entrenamiento modelo con todas las variables...')
        model = GridSearchCV(base_model, grid, cv = 5, n_jobs = -1, verbose = 3)
        model.fit(X_train_std, y_train)
        model_name = str(model.__class__).replace("'>", '').split('.')[-1]
        
        print('Resultados grilla')
        print("\n The best estimator across ALL searched params:\n",model.best_estimator_)
        print("\n The best score across ALL searched params:\n",model.best_score_)
        print("\n The best parameters across ALL searched params:\n",model.best_params_)
        print("\n The coeficiente de determinación  de la predicción across ALL searched params:\n",model.score(X_test_std, y_test))
        
        model = model.best_estimator_
        #pickle.dump(model, open(f"models/{model_name}{'' if con_outliers else '_so'}.sav",'wb'))  
        pickle.dump(model, open(path1, 'wb'))
        
        print('Comienza identificación de top n variables relevantes...')
        plt.figure(figsize = (10, 10))
        variables_relevantes = plot_importance(model, X_train.columns, n = 30)
        plt.tight_layout()
        plt.show()
        
        print('Comienza entrenamiento modelo variables relevantes...')
        X_train_dep_std, X_test_dep_std = get_scaled_matrix(X_train, X_test, variables_relevantes)
        
        model_dep =  GridSearchCV(base_model, grid, cv = 5, n_jobs = -1, verbose = 3)
        model_dep.fit(X_train_dep_std, y_train)
        
        print('Resultados grilla modelo depurado')
        print("\n The best estimator across ALL searched params:\n",model_dep.best_estimator_)
        print("\n The best score across ALL searched params:\n",model_dep.best_score_)
        print("\n The best parameters across ALL searched params:\n",model_dep.best_params_)
        print("\n The coeficiente de determinación  de la predicción across ALL searched params:\n",model_dep.score(X_test_dep_std, y_test))
        model_dep = model_dep.best_estimator_
        pickle.dump(model_dep, open(path2, 'wb'))
        
    elif fit_or_load == 'load':
        print('Cargando modelos guardados...')      
        
        model = pickle.load(open(path1, 'rb'))
        
        plt.figure(figsize = (10, 10))
        variables_relevantes = plot_importance(model, X_train.columns, n = 30)
        plt.tight_layout()
        plt.show()
        
        X_train_dep_std, X_test_dep_std = get_scaled_matrix(X_train, X_test, variables_relevantes)
        
        model_dep = pickle.load(open(path2, 'rb'))
        
    else:
        print('Error')
    
    y_hat = model.predict(X_test_std)
    print('\nResultados modelo con todas las variables...')
    resultados_modelo(model, X_test_std, y_test, y_hat)
    df_temp = pd.DataFrame(y_hat).rename(columns={0: model_name1[:-4]})

    y_hat_dep = model_dep.predict(X_test_dep_std)
    print('\nResultados modelo depurado.')
    resultados_modelo(model_dep, X_test_dep_std, y_test, y_hat_dep)
    df_temp_dep = pd.DataFrame(y_hat_dep).rename(columns={0: model_name2[:-4]})
    
    df_pred = pd.concat([df_temp, df_temp_dep], axis = 1)
    
    return df_pred, variables_relevantes


def calcular_error_modelo(df_an, nombre_modelo, variable_group):
    """
    La función toma tres parámetros de entrada:
    df_an: Es un dataframe que contiene los datos utilizados para evaluar el modelo.
    nombre_modelo: Es una cadena que contiene el nombre del modelo.
    variable_group: Es una cadena que indica el nombre de la columna por la cual se deben agrupar los datos en el dataframe.
    La función realiza lo siguiente:
    Primero, se seleccionan las columnas del dataframe que se necesitan para el cálculo del error y se crea una copia del dataframe.
    Luego, se agrega una columna 'Error' al dataframe que contiene el error absoluto entre el valor predicho por el modelo y el valor real. También se agrega una columna 'Error2' que contiene el cuadrado del error absoluto.
    Se agrupan los datos por la columna especificada en variable_group y se calculan la mediana del error absoluto y el promedio del error cuadrático.
    Se crea un nuevo dataframe a partir de los resultados obtenidos y se transpone para que las columnas del dataframe sean los grupos y las filas sean los valores de error.
    Se renombran las columnas del dataframe y se seleccionan sólo las filas que contienen los valores de error.
    Finalmente, se renombran las filas del dataframe y se devuelve el dataframe resultante.
    
    """
    columnas_mantener = [nombre_modelo, variable_group, 'total_minutes'] 
    df_temp = df_an.copy()[columnas_mantener]
    df_temp['Error'] = np.abs(df_temp[nombre_modelo] - df_temp['total_minutes'])
    df_temp['Error2'] = np.square(df_temp['Error'])
    df_error_grouped = df_temp.groupby([variable_group]).agg(
            {
                'Error': np.median, 
                'Error2': np.mean
            }
        ).reset_index()

    df_error_grouped.columns = [variable_group, 'ErrorMedianoAbsoluto', 'ErrorCuadraticoMedio']
    df_error_grouped = df_error_grouped.transpose()
    df_error_grouped.columns = df_error_grouped.loc[variable_group].astype(object)
    df_error_grouped = df_error_grouped.loc['ErrorMedianoAbsoluto': 'ErrorCuadraticoMedio']
    df_error_grouped.index = nombre_modelo + '_' + np.array(df_error_grouped.index)
    return df_error_grouped


def segmentation_model(df_aux, models_list, var_group, grid_list, nombre_modelo):
    """
    Entrena varios modelos en el conjunto de datos especificado, segmentado por la variable `var_group`, y devuelve un dataframe con los resultados de prueba.
    
    Parameters:
    df_aux (pd.DataFrame): El conjunto de datos a utilizar para el entrenamiento y la prueba.
    models_list (List[Type[BaseEstimator]]): La lista de modelos a entrenar.
    var_group (str): La variable por la cual se segmentará el conjunto de datos.
    grid_list (List[Dict]): La lista de grillas de parámetros a utilizar para el ajuste de cada modelo.
    nombre_modelo (str): El nombre a utilizar para identificar a cada modelo en el dataframe de resultados.
    
    Returns:
    pd.DataFrame: Un dataframe con dos columnas, 'order_id' y 'nombre_modelo', con los resultados de prueba para cada modelo.
    """
    
    df_final = pd.DataFrame(data = None, columns = ['order_id', nombre_modelo])
    
    casos_posibles = np.sort(df_aux[var_group].unique())
    
    if len(casos_posibles) != len(models_list) or len(casos_posibles) != len(grid_list):
        print('Error')
        return
        
    X = df_aux.copy()
    y = X.pop('total_minutes')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    X_train = X_train.drop(columns = ['order_id'])
    X_train.head()
    
    orders_test = pd.DataFrame(X_test['order_id']).rename(
        columns={0: 'order_id'}).reset_index(drop = True)
    
    for model, var, grid in zip(models_list, casos_posibles, grid_list):
        X_train_aux = X_train[X_train[var_group] == var].copy()
        X_test_aux  =  X_test[X_test[var_group] == var].copy()
        y_train_aux = y_train[X_train[var_group] == var].copy()
        y_test_aux  = y_test[X_test[var_group] == var].copy()
        
        orders_test_aux = pd.DataFrame(X_test_aux.pop('order_id')).rename(columns={0: 'order_id'}).reset_index(drop = True)      
        
        scaler = MinMaxScaler().fit(X_train_aux)
        X_train_aux = scaler.transform(X_train_aux)
        X_test_aux = scaler.transform(X_test_aux)

        model_aux = GridSearchCV(model,grid, cv = 5, n_jobs = -1, verbose = 3)
        model_aux.fit(X_train_aux, y_train_aux)
        
        y_hat_aux = model_aux.predict(X_test_aux)
        print(f'Los resultados para la variable {var_group} = {var} son:')
        resultados_modelo(model_aux, X_test_aux, y_test_aux, y_hat_aux)
        
        df_pred_aux = pd.DataFrame(y_hat_aux).rename(columns={0: nombre_modelo})
        orders_test_aux = pd.concat([orders_test_aux, df_pred_aux], axis = 1)
        
        df_final = pd.concat([
            df_final, orders_test_aux
        ], axis = 0)
        
        print('Comienza identificación de top n variables relevantes...')
        plt.figure(figsize = (10, 10))
        variables_relevantes = plot_importance(model_aux.best_estimator_, X_train.columns, n = 25)
        plt.tight_layout()
        plt.show()
        
        #orders_test = pd.merge(orders_test, orders_test_aux,
        #                      how = 'left', left_on = 'order_id', right_on = 'order_id') 
        print('')
        
    return df_final