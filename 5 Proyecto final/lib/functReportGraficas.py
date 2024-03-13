#!/usr/bin/env python
# -*- coding: utf-8 -*-



from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def reportar_error(y_pred, y_test):
    mse = mean_squared_error(y_pred, y_test)
    rmse = np.sqrt(mse)
    medianerror = median_absolute_error(y_pred, y_test)
    
    print(f'El error cuadrático medio es {mse:.2f}, mientras que su raiz es {rmse:.2f}')
    print(f'El error absoluto mediano es {medianerror:.2f}')
    
    
def resultados_modelo(modelo, X_test_aux, y_test_aux, y_hat, print_r2 = False):
    """
        Calcula y reporta tres medidas de error para un conjunto de predicciones y valores reales. También puede imprimir el coeficiente de determinación (r2) si se especifica el argumento opcional `print_r2`.
        
        Parámetros:
        modelo (modelo): El modelo de aprendizaje automático utilizado para hacer las predicciones.
        X_test_aux (ndarray): Un conjunto de datos de prueba utilizado para evaluar el modelo.
        y_test_aux (ndarray): Un vector de valores reales para el conjunto de datos de prueba.
        y_hat (ndarray): Un vector de predicciones hechas por el modelo.
        print_r2 (bool, opcional): Si es True, se imprime el coeficiente de determinación (r2). Por defecto es False.
        
        Returns:
        None: La función no retorna ningún valor.
    """
    
    mse = mean_squared_error(y_hat, y_test_aux)
    rmse = np.sqrt(mse)
    medianerror = median_absolute_error(y_hat, y_test_aux)
    print(f'El error cuadrático medio es {mse:.2f}, mientras que su raiz es {rmse:.2f}')
    print(f'El error absoluto mediano es {medianerror:.2f}')
    
    if print_r2:
        result = modelo.score(X_test_aux, y_test_aux)
        print("\n El coeficiente de determinación de la predicción es: ",result, "\n Para que sea bueno tiene que ser cercano a 1")
    
    
    
    
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
        tmp_importance = fit_model.coef_
        
        titulo = 'Coeficientes'
    else:
        return 'Solo soporta modelos que tienen atributos coef_ o feature_importances_'
    tmp_importance_abs = np.abs(tmp_importance)
    sort_importance = np.argsort(tmp_importance_abs)[::-1][:n] # Retorna un array con n índices más importantes
    names = [feat_names[i] for i in sort_importance] # obtiene los nombres de las columnas en las posiciones
    plt.title(titulo)
    plt.barh(range(n), list(tmp_importance[sort_importance])[::-1]) # Grafica los valores de las columnas más importantes
    plt.yticks(range(n), names[::-1], rotation=0)
    return names
