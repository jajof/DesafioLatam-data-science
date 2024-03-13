# Control de versiones
&nbsp;

| Fecha  | Versión | Cambios de la versión anterior | Actualizado por | Revisado por | Aprobado por |
|---|---|---|---|---|---|
| 15-12-2022  | 1.1  | Versión inicial  | MGS  |  |  |
&nbsp;
# Estructura de directorio
&nbsp;
A continuacion, se muestra la estructura base de este directorio
&nbsp;

| **objeto**   |  **Tipo**  | **Descripción** |
|--------------|------------|-------------|
| data         | Directorio | El directorio de datos almacena todos los archivos de datos. |
| data\input   | Directorio | Se utiliza para almacenar todos los archivos de datos sin procesar. |
| data\output  | Directorio | Se utiliza para almacenar todos los archivos de datos generados producto de un proceso de análisis o producción. |
| docs         | Directorio | Aquí puede almacenar cualquier fuente o material de referencia sobre el proyecto. |
| image        | Directorio | Se utiliza para almacenar imágenes. |
| image\input  | Directorio | Se utiliza para almacenar imágenes que serán utilizadas para informes. |
| image\output | Directorio | Se utiliza para almacenar imágenes generadas producto de un proceso de análisis o producción. |
| lib          | Directorio | Se utiliza para almacenar funciones personalizadas. Los scripts proporcionan una funcionalidad útil, pero no constituyen scripts de análisis o producción. |
| models       | Directorio | Se utiliza para almacenar el resultado de los modelos |
| ppt          | Directorio | Se utiliza para almacenar presentacion relativa al proyecto. |
| \*.ipynb     | Archivo    | Script análisis o producción Jupyter. |
| \*.py        | Archivo    | Script análisis o producción Python. |
| readme.md    | Archivo    | El archivo de introducción se utiliza para proporcionar orientación al proyecto.  El archivo define los objetivos del proyecto y está destinado a presentar datos del proyecto, código fuente y configuraciones para una investigación repetible.  El formato de archivo * .md es el de un archivo de texto básico o de rebajas. Cuando se guarda en GitHub, se utilizará para crear un wiki de proyecto HTML. |

&nbsp;

**documento_tecnico.ipynb:** jupyter notebook principal con información consolidada de los 5 notebook particulares.

&nbsp;

# Árbol de directorio
```
la_promesa/
├── data/
│   ├── input/
│   │   ├── 1_Orden_compra.csv
│   │   ├── 2_Orden_productos.csv
│   │   ├── 3_Recolector.csv
│   │   └── 4_tienda.csv
│   └── output/
│       ├── dataset_full_data.csv
│       ├── dataset_full_data.plk
│       ├── dataset_full_data.xlsx
│       ├── dataset_pre_modelacion.csv
│       ├── dataset_pre_modelacion.plk
│       ├── dataset_pre_modelacion.xlsx
│       ├── product_order.plk
│       ├── purchase_order.plk
│       ├── shopper.plk
│       └── store.plk
├── docs/
│   ├── DataAnalytics.vsdx
│   ├── modelo.dio
│   ├── Resumen_Ejecutivo_LaPromesa_ConsultoraFocus.docx
│   └── Resumen_Ejecutivo_LaPromesa_ConsultoraFocus.pdf
├── image/
│   ├── input/
│   │   ├── diagrama.png
│   │   ├── modelo.png
│   │   └── skimpy.png
│   └── output/
├── lib/
│   ├── functAnalyticsHelpers.py
│   ├── functChecModule.py
│   ├── functCleanData.py
│   ├── functDescriptiveAnalysis.py
│   ├── functML.py
│   ├── functReportGraficas.py
│   ├── functTransformData.py
│   └── readme.md
├── models/
│   ├── ElasticNetCV.sav
│   ├── ElasticNetCV_dep.sav
│   ├── ElasticNetCV_dep_so.sav
│   ├── ElasticNetCV_so.sav
│   ├── GradientBoostingRegressor.sav
│   ├── GradientBoostingRegressor_dep.sav
│   ├── GradientBoostingRegressor_dep_so.sav
│   ├── GradientBoostingRegressor_so.sav
│   ├── LinearRegression.sav
│   ├── LinearRegression_dep.sav
│   ├── LinearRegression_dep_so.sav
│   ├── LinearRegression_so.sav
│   ├── RandomForestRegressor.sav
│   ├── RandomForestRegressor_dep.sav
│   ├── RandomForestRegressor_dep_so.sav
│   ├── RandomForestRegressor_so.sav
│   ├── SGDRegressor.sav
│   ├── SGDRegressor_dep.sav
│   ├── SGDRegressor_dep_so.sav
│   └── SGDRegressor_so.sav
├── ppt/
│   ├── hito_4_La_Promesa.pdf
│   └── hito_4_La_Promesa.pptx
├── readme.md
├── 00_notebookMaster.ipynb
├── 01_AnalisisDescriptivoPreeliminar.ipynb
├── 02_cleanTransform.ipynb
├── 03_joinDataset.ipynb
├── 04_Feature_Engineering_AnalisisDescriptivo.ipynb
├── 05_Modelacion.ipynb
├── 06_Segmentacion.ipynb
└── documento_tecnico.ipynb
```
