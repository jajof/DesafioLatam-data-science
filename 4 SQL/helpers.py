#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle, datetime, os, glob
import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path


def train_and_pickle(model, X_train, y_train):
    """
        Dado un modelo sklearn, además de X_train e y_train, se entrena el modelo
        y luego se genera un archivo serializado con un nombre identificable.
        
        model: sklearn model class
        X_train: matriz variables independientes
        y_train: vector objetivo
    """
    # Fiteo un modelo con X_train e y_train
    tmp_model_train = model.fit(X_train, y_train)
    # Extraigo el nombre del modelo
    model_name = str(model.__class__).replace("'>", '').split('.')[-1]
    # El nombre del archivo debe ser el VectorObjetivo_Modelo.sav
    nombre = f"{y_train.name}_{model_name}.sav"
    # Se guardará en la ruta ./Modelos
    path = Path("./Modelos")  / nombre
    pickle.dump(tmp_model_train, open(path, 'wb'))



def create_grouped_probabilty(model, X_test, vector_objetivo, variables):
    """Retorna un DataFrame agrupado por variables que para cada caso tiene la probabilidad
    predicha para la variable de interés del modelo.
    
    model: sklearn model class
    X_test: matriz de variables independientes
    vector_objetivo: string que indica el nombre del vector objetivo
    variables: lista de variables de X_test para el group by
    
    return:
        DataFrame
    """
    tmp = X_test.copy()
    tmp_pr = model.predict_proba(X_test) 

    df_prob = pd.concat([
        tmp.reset_index(drop = True), 
        pd.DataFrame(tmp_pr[:, 1], columns = [vector_objetivo])],
        axis = 1)
    
    tmp_query = df_prob.groupby(variables)[vector_objetivo].mean()
    return tmp_query



