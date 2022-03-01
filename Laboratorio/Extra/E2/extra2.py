"""
EXTRA 2: COMPLEJIDAD DE COMPUTACIÓN
Martín Fernández de Diego
"""

import os
import numpy as np
import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt

"""
Dado un dataframe
devuelve su índice de Gini
"""
def ind_gini(distr):
    # Calculamos la frecuencia acumulada
    probab_acumulada = np.empty([len(distr['probab'])])
    probab_acumulada[0] = distr.at[0,'probab']
    for i in range(1,len(probab_acumulada)):
        probab_acumulada[i] = probab_acumulada[i-1]+distr.at[i,'probab']
    # Mostramos la gráfica de frecuencia acumulada
    plt.plot(np.linspace(0,1,len(probab_acumulada)),probab_acumulada)
    plt.plot(np.linspace(0,1,len(probab_acumulada)),np.linspace(0,1,len(probab_acumulada)))
    # Calculamos el índice de Gini
    sum = 0
    for i in range(1,len(probab_acumulada)):
        sum += (probab_acumulada[i]+probab_acumulada[i-1])/len(probab_acumulada)
    return 1-sum

"""
Dado un dataframe
devuelve su diversidad ^2D de Hill
"""
def div_2dhill(distr):
    h = 0
    for p in distr['probab']:
        h += p*p
    return 1/h


# FORMATO
class Formato:
    BOLD = "\033[1m"
    RESET = "\033[0m"

#### Vamos al directorio de trabajo####
os.getcwd()
#os.chdir(ubica)
#files = os.listdir(ruta)

with open('GCOM2022_pract2_auxiliar_num.txt', 'r',encoding="utf8") as file:
      num = file.read()

with open('GCOM2022_pract2_auxiliar_eng.txt', 'r',encoding="utf8") as file:
      en = file.read()

tab_num = Counter(num)
##### Transformamos en formato array de los carácteres (states) y su frecuencia
##### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_num_states = np.array(list(tab_num))
tab_num_weights = np.array(list(tab_num.values()))
tab_num_probab = tab_num_weights/float(np.sum(tab_num_weights))
distr_num = pd.DataFrame({'states': tab_num_states, 'probab': tab_num_probab})
distr_num = distr_num.sort_values(by='probab', ascending=True)
distr_num.index=np.arange(0,len(tab_num_states))

# APARTADO i)
print("\n" + Formato.BOLD + "Apartado i)" + Formato.RESET)
print("Índice de Gini de {0,1,0,0,0,2,1,1,0,2,0,1,2,0,2,0,1,1}:", ind_gini(distr_num))
print("Diversidad ^2D de Hill de {0,1,0,0,0,2,1,1,0,2,0,1,2,0,2,0,1,1}:", div_2dhill(distr_num))
plt.show()

tab_en = Counter(en)
##### Transformamos en formato array de los carácteres (states) y su frecuencia
##### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))

# APARTADO ii)
print("\n" + Formato.BOLD + "Apartado ii)" + Formato.RESET)
print("Índice de Gini de S_english:", ind_gini(distr_en))
print("Diversidad ^2D de Hill de S_english:", div_2dhill(distr_en))
plt.show()
