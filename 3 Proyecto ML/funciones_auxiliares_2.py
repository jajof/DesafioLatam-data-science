#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')


## Helpers


def infer_datatype(df, datatype, drop_none=True):
    """ A partir de un dataset y un tipo de datos entregado, devuelve los nombres de las columnas
        del dataset que tienen el correspondiente tipo de dato.
        
        Argumentos:
           - df: Dataframe de pandas.
           - datatype: String con el tipo de dato que se desea consultar a las columnas del dataframe.
           - drop_none: Filtra las columnas cuyo tipo de dato no esté especificado. default = True.
    """
    tmp_list = [i if df[i].dtype == datatype else None for i in df.columns]
    if drop_none is True:
        tmp_list = list(filter(lambda x: x != None, tmp_list))

    return tmp_list

def return_time_string(var, date_format='%m%d%Y'):
    return var.apply(lambda x: datetime.strptime(str(x), date_format))

def count_freq(df, selected_columns):
    """ Cuenta la cantidad de valores únicos y la frecuencia de dichos valores en las columnas
        entregadas por `selected_columns`.
        
        Argumentos:
            - df: dataframe que contiene las columnas en cuestión.
            - selected_columns: Columnas del dataframe de las que se quiere saber la frecuencia de valores.
    """
    return {i: df[i].unique().shape[0] for i in selected_columns}


def create_suitable_dataframe(df):
    """TODO: Crea un dataframe apto para entrenamiento de acuerdo a normas básicas de limpieza de datos faltantes,
        transformación de etiquetas nulas en variables categóricas y crea atributos sinteticos de edad del sospechoso
         y conversión de distancia a sistema metrico.
    Argumentos:
        - df: Un objeto pandas.DataFrame 
    returns: 
    """
    ### Obtener columnas por tipo de dato
    object_data_type = infer_datatype(df, 'object')
    integer_data_type = infer_datatype(df, 'int64')
    float_data_type = infer_datatype(df, 'float')
    
    # Quiero recuperar la lista de valores numericos tambien
    suitable_numerical_attributes = list(integer_data_type) + list(float_data_type)
    print(suitable_numerical_attributes)
    
    ### Contar la cantidad de clases en el caso de las var. categóricas y frecuencia de valores para las numéricas
    object_unique_vals = count_freq(df, object_data_type)
    int_unique_vals = count_freq(df, integer_data_type)
    float_unique_vals = count_freq(df, float_data_type)
    
    ### Selección de atributos categoricos que cumplen con características deseadas
    suitable_categorical_attributes = dict(filter(lambda x: x[1] < 100 and x[1] >= 2, object_unique_vals.items()))
    suitable_categorical_attributes = list(suitable_categorical_attributes.keys())

    ### Reemplazo de clases faltantes
    ### {N: No, Y: Yes, U: Unknown}
    df['officrid'] = np.where(df['officrid'] == ' ', 'N', 'Y')
    df['offshld'] = np.where(df['offshld'] == ' ', 'N', 'Y')
    df['sector'] = np.where(df['sector'] == ' ', 'U', df['sector'])
    df['trhsloc'] = np.where(df['trhsloc'] == ' ', 'U', df['trhsloc'])
    df['beat'] = np.where(df['beat'] == ' ', 'U', df['beat'])
    df['offverb'] = np.where(df['offverb'] == ' ', 'N', 'Y')
    
    meters = df['ht_feet'].astype(str) + '.' + df['ht_inch'].astype(str)
    df['meters'] = meters.apply(lambda x: float(x) * 0.3048) # Conversión de distanca a sistema metrico (non retarded)
    df['month'] = return_time_string(df['datestop']).apply(lambda x: x.month) # Agregación a solo meses
    
    ### Calculo de la edad del suspechoso
    age_individual = return_time_string(df['dob']).apply(lambda x: 2009 - x.year)
    # Filtrar solo mayores de 18 años y menores de 100
    df['age_individual'] = np.where(np.logical_and(df['age'] > 18, df['age'] < 100), df['age'], np.nan)
    proc_df = df.dropna()
    preserve_vars = suitable_categorical_attributes + ['month', 'meters']
    proc_df = proc_df.loc[:, preserve_vars] # Agregar los atributos sintéticos al df
    return proc_df, suitable_categorical_attributes, suitable_numerical_attributes

def create_suitable_dataframe_with_nan(df):
    """
    Crea un dataframe apto para entrenamiento de acuerdo a normas básicas de limpieza de datos faltantes,
    transformación de etiquetas nulas en variables categóricas y crea atributos sinteticos de edad del sospechoso
    y conversión de distancia a sistema metrico.
    
    Argumentos:
        - df: Un objeto pandas.DataFrame 
    
    returns: 
        - proc_df: Dataframe de salida.
        - suitable_categorical_attributes: listado de variables categoricas.
        - suitable_numerical_attributes: lista con varoiables numericas.
    """
    #
    # Obtener columnas por tipo de dato
    #
    object_data_type = infer_datatype(df, 'object')
    integer_data_type = infer_datatype(df, 'int64')
    float_data_type = infer_datatype(df, 'float')
    #
    # Recuperacion de atributos numericos
    #
    suitable_numerical_attributes = list(integer_data_type) + list(float_data_type)
    suitable_numerical_attributes.remove("ht_feet")
    suitable_numerical_attributes.remove("ht_inch")
    
    #
    # Contar la cantidad de clases en el caso de las var. categóricas y frecuencia de valores para las numéricas
    #
    object_unique_vals = count_freq(df, object_data_type)
    int_unique_vals = count_freq(df, integer_data_type)
    float_unique_vals = count_freq(df, float_data_type)
    
    #
    # Selección de atributos categoricos que cumplen con características deseadas
    #
    suitable_categorical_attributes = dict(filter(lambda x: x[1] < 100 and x[1] >= 2, object_unique_vals.items()))
    suitable_categorical_attributes = list(suitable_categorical_attributes.keys())

    #
    # Reemplazo de clases faltantes, para el caso de las desconocidas las dejamos como np.nan
    #
    # N: No
    # Y: Yes
    # U: np.nan
    #
    df['officrid'] = np.where(df['officrid'] == ' ', 'N', 'Y')
    df['offshld'] = np.where(df['offshld'] == ' ', 'N', 'Y')
    df['sector'] = np.where(df['sector'] == ' ', np.nan, df['sector'])
    df['trhsloc'] = np.where(df['trhsloc'] == ' ', np.nan, df['trhsloc'])
    df['offverb'] = np.where(df['offverb'] == ' ', 'N', 'Y')
    
    #
    # En este caso, las recodificamos y las transformamos a valor numerico.
    # Se utiliza float debido al formato del np.nan
    #
    df['beat'] = np.where(df['beat'].isin([' ', "U"]), np.nan, df['beat'])
    df['beat'] = df['beat'].astype(float)
    
    #
    # Igual al caso anterior, recodificamos y transformamos a variable numerica.
    #
    df['post'] = np.where(df['post'] == ' ', np.nan, df['post'])
    df['post'] = df['post'].astype(float)
    
    #
    # Transformacion de longitiud a metro.
    #
    meters = df['ht_feet'].astype(str) + '.' + df['ht_inch'].astype(str)
    df['meters'] = meters.apply(lambda x: float(x) * 0.3048)
    
    #
    # Solo mantenemos los meses.
    #
    df['month'] = return_time_string(df['datestop']).apply(lambda x: x.month)
    
    # 
    # Filtramos solo mayores de 18 años y menores de 100.
    #
    df['age'] = np.where(np.logical_and(df['age'] > 18, df['age'] < 100), df['age'], np.nan)
    
    #
    # Filtramos los que pesen mas de 6 libras y menos de 999 
    #
    df['weight'] = np.where(np.logical_and(df['weight'] > 6, df['weight'] < 999), df['weight'], np.nan)
    
    #
    # Seleccionamos las variables que mantendremos en el dataframe de salida.
    #
    preserve_vars = suitable_categorical_attributes + ['month', 'meters'] + suitable_numerical_attributes
    
    #
    # Las agregamos al dataframe
    #
    proc_df = df.loc[:, preserve_vars]
    
    #
    # Retornamos:
    #     proc_df: Dataframe de salida.
    #     suitable_categorical_attributes: listado de variables categoricas.
    #     suitable_numerical_attributes: lista con varoiables numericas.
    #
    return proc_df, suitable_categorical_attributes, suitable_numerical_attributes


def graficar(df, layout, hue, value, figsize):
    """
    Grafica una grilla de datos de df con un layout entregado como tupla. Divide la data
    según hue y grafica las frecuencias de value.
    
    Argumentos:
        - df: Un objeto pandas.DataFrame 
        - df: tupla de dos posiciones
        - hue: string que hace referencia a una columna categórica del df
        - value: columna cuyos valores se contarán
    
    returns: 
        -None
    """
    plt.figure(figsize = figsize)
    for i, r in enumerate(df[hue].unique()):
        plt.subplot(layout[0], layout[1], i + 1)
        df_temp = df[df[hue] == r].copy()
        ax = sns.countplot(
            x = df_temp[value],                          
            order = df_temp[value].value_counts().index,
            palette="Blues_r")
        for p, label in zip(ax.patches, df_temp[value].value_counts('%')):
            ax.annotate(np.round(label, 2), (p.get_x()+0.375, p.get_height()+1))
        plt.title(f'Frecuencia {value} para {hue} {r}')
    plt.tight_layout()


def plot_metrics(reports=[], labels=[], figsize=(18,5), xticks=np.linspace(0, 1, 6).round(1)):
    """
        Genera un DataFrame con las métricas de classification report entregadas, y plotea las métricas
        @params:
            reports: Requerido. Lista de diccionarios de "classification reports". Cada diccionario debe tener las métricas
            'f1', 'recall' y 'precision' para todas las clases
            labels: Requerido. Lista con las etiquetas de cada modelo. Debe haber la misma cantidad de etiquetas que de reportes.
            figsize: Opcional, por defecto (18,5). Tupla con dimensiones del lienzo de gráficos.
            xticks: Opcional, por defecto [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]. Lista con límite inferior y límite superior para eje x de los gráficos.
        @return:
            df: Retorna un DataFrame generado con las métricas F1, precision, recall como columnas, 
            y cada reporte ingresado como fila. Si no se pudo generar un DataFrame, se imprime un error, y retorna False.
    """
    
    if len(reports) == len(labels):
        metrics = pd.concat([
            pd.DataFrame([
                {
                    **{"Clase/Pond": f"Clase {j}".title() if j.isdigit() else j.title()},
                    **reports[i][j],
                    **{"Modelo": labels[i]}
                }
                for j in reports[i].keys()
                if j != "accuracy"
            ])
            for i in range(len(reports))
        ])
        
        if metrics is not None:
            met_melt = metrics.drop(columns=["support"]).melt(id_vars=["Clase/Pond", "Modelo"]).round(2)
            
            metric_labels = met_melt["variable"].unique()
            fig, axes = plt.subplots(1, len(metric_labels), figsize=figsize, sharey=True)
            fig.suptitle("Métricas de desempeño", fontsize=16)

            for i in range(len(axes)):
                ss = met_melt[met_melt["variable"]==metric_labels[i]]
                sns.barplot(ax=axes[i], y="Clase/Pond", x="value", hue="Modelo", data=ss, palette=sns.color_palette("Set2", 5));
                axes[i].legend([],[], frameon=False)    

                if i > 0:
                    axes[i].set_ylabel("")     

                for container in axes[i].containers:
                    axes[i].bar_label(container, fontsize=10)

                axes[i].set_xlabel(metric_labels[i].title());
                axes[i].set_xticks(xticks);

            axes[1].legend(loc="upper left", bbox_to_anchor=(-0.5,1.12), ncol=len(met_melt["Modelo"].unique()));
            
            return metrics
        else:
            print("No se pudo generar un DataFrame!")
            return False
        
    else:
        print("Debe ingresar misma cantidad de reportes y de etiquetas!")
        return False
    
    
    
def plot_importance(fit_model, feat_names, n=10):
    """
        Genera un gráfico con los atributos más importantes de un modelo
        @params:
            fit_model: Requerido. Modelo entrenado. De momento acepta Logistic Regression, Gradient Boosting y Random Forest Regressor
            feat_names: Requerido. Lista con las columnas usadas en el entrenamiento
            n: Opcional, top de columnas que se desean extraer
        @return:
            list: Retorna una lista con las columnas más importantes
    """
    
    if hasattr(fit_model, 'feature_importances_'):
        tmp_importance = fit_model.feature_importances_
        titulo = 'Feature importance'
    elif hasattr(fit_model, 'coef_'):
        tmp_importance = fit_model.coef_[0]
        titulo = 'Coeficientes'
    else:
        return 'Solo soporta modelos que tienen atributos coef_ o feature_importances_'
    sort_importance = np.argsort(tmp_importance)[::-1][:n] # Retorna un array con n índices más importantes
    names = [feat_names[i] for i in sort_importance] # obtiene los nombres de las columnas en las posiciones
    plt.title(titulo)
    plt.barh(range(n), list(reversed(tmp_importance[sort_importance]))) # Grafica los valores de las columnas más importantes
    plt.yticks(range(n), names, rotation=0)
    return names