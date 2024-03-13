# Control de versiones
&nbsp;

| Fecha  | Versión | Cambios de la versión anterior | Actualizado por | Revisado por | Aprobado por |
|---|---|---|---|---|---|
| 01-08-2022  | 1.1  | Versión inicial  | MGS  |  |  |
&nbsp;
# Estructura de directorio
&nbsp;
A continuacion, se describen brevemente las funciones en este directorio
&nbsp;

| **archivo**       | **funcion** | **Descripción** |
|--------------|-----|-------------|
| functElastiScroll.py        | elasticScroll | Realiza consultas a indice elasticsearch sin limitaciones por el numero de documentos devuelto |
| functCalcFormatTime.py      | formatDate | realiza calculos de tipos de fecha para consumo en consultas, formateo de nombre de archivos, etc. |
| functCleanData.py           | dropUnnamedCols | tiene por objetivo quitar las columnas que comienzan unnamed |
|                             | is_empty | tiene por objetivo validar si la estructura de datos preguntada contiene datos o esta vacia |
|                             | normalizeColNames | tiene por objetivo normalizar nombre de las columnas de un dataframe |
|                             | dropNaN | tiene por objetivo eliminar valores perdidos de un dataframe pudiendo eliminar columnas o filas |
| functChecModule.py          | importOrInstall | tiene por objetivo validar si el modulo existe y en caso contrario instalar dicho modulo. |
| functDescriptiveAnalysis.py | PlotHistDA | tiene por objetivo generar histogramas de distintas columnas de un dataframe, incluyendo lineas verticales para representar la media y mediana |
|                             | inspectDataFrameDA | tiene por objetivo generar un analisis descriptivo de distintas columnas de un dataframe, separando el analisis entre variables de tipo numerico y string |
|                             | countNaNDA | tiene por objetivo generar un cuadro resumen del total de valores perdidos y su respectivo porcentaje |
|                             | PlotCorrDA | tiene por objetivo generar un grafico de correlacion de distintas columnas de un dataframe comparando una columna en particular |
|                             | contentAnalysisDA | tiene por objetivo generar un cuadro resumen del % de los valores contenidos en cada columna discreta ( o no continua |
|                             | countPlotDA | tiene por objetivo generar histogramas de distintas columnas de un dataframe, incluyendo lineas verticales para representar la media y mediana |
|                             | countPlotDescriptionDA | tiene por objetivo generar histogramas de distintas columnas de un dataframe, incluyendo lineas verticales para representar la media y mediana |
| functML.py                  | plotCurveROC | tiene por objetivo generar una grafica de la curva ROC |
|                             | calcSignificantVars | tiene por obtener las variables significativas de un modelo de acuerdo a un pvalue propuesto (por defecto se utilizara 0.05) |
| functTransformData.py       | ageRange | tiene por objetivo categorizar la edad |
|                             | binarizeDf | tiene por objetivo binarizar columnas de un dataframe |
|                             | extractNums | tiene por objetivo extraer numeros de un string |
# Árbol de directorio
```
cencosud/
└── lib/
    ├── readme.md
    ├── functML.py
    ├── functDescriptiveAnalysis.py
    ├── functChecModule.py
    ├── functCleanData.py
    ├── functCalcFormatTime.py
    └── functElastiScroll.py
```