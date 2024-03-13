# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:49:57 2022

@author: jahof
"""

import re
import string
from nltk import pos_tag, word_tokenize # Necesario para preprocesamiento
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # Lematizador para preprocesar texto
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn")
sns.set(font_scale=1.5)


def text_preprocessor(text, lmtz = WordNetLemmatizer()):
    '''
        Retorna un string a partir de uno entregado. El objetivo es preprocesarlo de cara a usarlo
        en análisis de texto. Además, cada palabra la retorna en su versión lematizada.
        
        Argumentos:
            - text: Es un string
            - lmtz: objeto WordNetLemmatizer() de ntlk.stem
        Returns:
            - Una versión preprocesada y lematizada del string    
    '''
    
    text = text.lower()
    text = re.sub('\[.*?¿\]\%', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…«»]', '', text)
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = re.sub('\n', ' ', text)
    
    def get_lemma(lmtz, word):
        tag_parts = pos_tag(word_tokenize(word))
        tag_letter = tag_parts[0][1][0].lower() # sacar la primera letra del tag
        tag_letter = tag_letter if tag_letter in ["a", "r", "n", "v"] else None
        return word if not tag_letter else lmtz.lemmatize(word, tag_letter)

    text = ' '.join([get_lemma(lmtz, word) for word in text.split()])
    return text





def get_words_freq_by_polarization(data, polarization, sw = stopwords.words("english"), lemmatizer = WordNetLemmatizer()):
    '''
        Retorna un DataFrame con las palabras y sus frecuencias para una lista de strings. En este caso,
        una lista de tweets. 
        
        Argumentos:
            - data: DataFrame que contiene los tweets en filas y los identifica como es_positivo = 1 o 0
            - polarization: 1 o 0 según se quieran ver tweets positivos o negativos
            - sw: lista de stopwords. Por defecto, se toman las de inglés
            - lemmatizer: objeto WordNetLemmatizer() de ntlk.stem
        Returns:
            - DataFrame con las palabras de los tweets y sus frecuencias.    
    '''
  
    vectorizer = TfidfVectorizer(
        analyzer="word", # Procesar palabras, no "caracteres"
        preprocessor=lambda x: text_preprocessor(x, lemmatizer),
        sublinear_tf=True, # google dice que sí
        min_df=5, # aumentar a 5 con más datos (cant. mínima de ocurrencias para preservar palabra, evita of)
        norm='l2', # norma euclídea de regularización (evita of, puede empeorar desempeño en train)
        encoding='latin-1', 
        ngram_range=(1, 2), # considerar palabras aisladas y pares de palabras
        stop_words= sw
    )
    
    vectorizer_fit = vectorizer.fit_transform(
                            data.loc[data['es_positivo'] == polarization, 'content'])
    
    words = vectorizer.get_feature_names_out()
    words_freq = vectorizer_fit.toarray().sum(axis = 0) # Sumo las frecuencias para cada palabra (columna)
    words_freq_df = pd.DataFrame([words, words_freq]).T.sort_values(by = [1], ascending = False)
    words_freq_df.columns = ["Palabra", "Frecuencia"]
    words_freq_df['Frecuencia'] = words_freq_df['Frecuencia']/np.sum(words_freq_df['Frecuencia']) 
    return words_freq_df






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