import pip

def importOrInstall(package):
    """# Autor: Mauricio Gomez S.
    # Fecha: 13-07-2022
    # Descripcion: La siguiente funcion tiene por objetivo validar si el modulo existe
    #              y en caso contrario instalar dicho modulo.
    #
    # Parametros: 
    #             package [str] nombre del modulo a evaluar si existe.
    #
    # Return: 
    #            df [dataframe] dataframe sin columnas que comiencen por Unnamed
    #
    # fuente: https://stackoverflow.com/questions/4527554/check-if-module-exists-if-not-install-it
    #"""
    try:
        __import__(package)
        message = 'el modulo ' + package + ' se encuentra instalado'
    except ImportError:
        pip.main(['install', package]) 
        message = 'el modulo ' + package + ' se instalo exitosamente'

#

#

#

#

#

#
