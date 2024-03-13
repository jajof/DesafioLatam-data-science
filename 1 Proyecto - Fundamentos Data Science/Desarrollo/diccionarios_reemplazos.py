import numpy as np

dict_occupation = {
        'Prof-specialty':'white-collar',
        'Exec-managerial':'white-collar',
        'Adm-clerical':'white-collar',
        'Sales':'white-collar',
        'Tech-support':'white-collar',
        'Craft-repair': 'blue-collar',
        'Machine-op-inspct' : 'blue-collar',
        'Transport-moving': 'blue-collar',
        'Handlers-cleaners' : 'blue-collar',
        'Farming-fishing': 'blue-collar',
        'Protective-serv': 'blue-collar',
        'Priv-house-serv': 'blue-collar',
        'Other-service': 'others',
        'Armed-Forces': 'others',
        np.nan: 'others'} 

dict_workclass = {
    'Federal-gov': 'federal-gov',
    'State-gov': 'state-level-gov',
    'Local-gov': 'state-level-gov',
    'Self-emp-inc': 'self-employed',
    'Self-emp-not-inc': 'self-employed',
    'Never-worked': 'unemployed',
    'Without-pay': 'unemployed',
    np.nan: 'others'}

dict_education = {
    'HS-grad'         : 'high-school', 
    'Some-college'    : 'college', 
    'Bachelors'       : 'university', 
    'Masters'         : 'university', 
    'Assoc-voc'       : 'college', 
    '11th'            : 'high-school', 
    'Assoc-acdm'      : 'college', 
    '10th'            : 'high-school', 
    '7th-8th'         : 'high-school', 
    'Prof-school'     : 'university', 
    '9th'             : 'high-school', 
    '12th'            : 'high-school', 
    'Doctorate'       : 'university', 
    '5th-6th'         : 'elementary-school', 
    '1st-4th'         : 'elementary-school', 
    'Preschool'       : 'preschool'}

dict_marital_status = {
        'Married-civ-spouse'    : 'married',
        'Divorced'              : 'divorced',
        'Separated'             : 'separated',
        'Widowed'               : 'widowed',
        'Married-spouse-absent' : 'married',
        'Married-AF-spouse'     : 'married'}


dict_native_country = {
        'United-States'             : 'America', 
        'Mexico'                    : 'America', 
        'Philippines'               : 'Asia', 
        'Germany'                   : 'Europa', 
        'Puerto-Rico'               : 'America', 
        'Canada'                    : 'America', 
        'El-Salvador'               : 'America', 
        'India'                     : 'Asia', 
        'Cuba'                      : 'America', 
        'England'                   : 'Europa', 
        'China'                     : 'Asia', 
        'South'                     : 'Asia', 
        'Jamaica'                   : 'America', 
        'Italy'                     : 'Europa', 
        'Dominican-Republic'        : 'America', 
        'Japan'                     : 'Asia', 
        'Guatemala'                 : 'America', 
        'Poland'                    : 'Europa', 
        'Vietnam'                   : 'Asia', 
        'Columbia'                  : 'America', 
        'Haiti'                     : 'America', 
        'Portugal'                  : 'Europa', 
        'Taiwan'                    : 'Asia', 
        'Iran'                      : 'Asia', 
        'Greece'                    : 'Europa', 
        'Nicaragua'                 : 'America', 
        'Peru'                      : 'America', 
        'Ecuador'                   : 'America', 
        'France'                    : 'Europa', 
        'Ireland'                   : 'Europa', 
        'Hong'                      : 'Asia', 
        'Thailand'                  : 'Asia', 
        'Cambodia'                  : 'Asia', 
        'Trinadad&Tobago'           : 'America', 
        'Yugoslavia'                : 'Europa', 
        'Outlying-US(Guam-USVI-etc)': 'America', 
        'Laos'                      : 'Asia', 
        'Scotland'                  : 'Europa', 
        'Honduras'                  : 'America', 
        'Hungary'                   : 'Europa', 
        'Holand-Netherlands'        : 'Europa',
        np.nan                      : 'Otro'
}