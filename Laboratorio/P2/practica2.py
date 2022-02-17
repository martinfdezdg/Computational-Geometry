"""
PRÁCTICA 2: CÓDIGO HUFFMAN
Belén Sánchez Centeno
Martín Fernández de Diego
"""

import os
import numpy as np
import pandas as pd
import math
from collections import Counter

"""
Dado un dataframe
devuelve una rama del arbol de Huffman
"""
def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
    probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab, })
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

"""
Dado un dataframe
devuelve un arbol de Huffman
"""
def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = np.array([branch['codigo']])
        tree = np.concatenate((tree, code), axis=None)
    return(tree)

"""
Dado un arbol
devuelve un diccionario con el codigo de cada caracter
"""
def extraer_cadena_caracter(tree):
    d = dict() # diccionario {carácter : código}   
    # Se recorre el árbol desde la raíz, que se encuentra en la última posición
    for i in range(tree.size-1,-1,-1):  
        # Se accede a ambas hojas
        for j in range(2): 
            estado = list(tree[i].items())[j][0]
            # Se sabe por construcción del árbol que en la primera hoja hay un 0 y en la segunda un 1
            codigo = str(j)
            # Se guardan o actualizan los caracteres en el diccionario
            for caracter in estado:
                if caracter in d:
                    d[caracter] += codigo # codigo (0 o 1)
                else:
                    d[caracter] = codigo # codigo (0 o 1)
    return d

"""
Dado un dataframe y un diccionario
devuelve la longitud media, es decir, la suma de las longitudes de los elementos por sus probabilidades
"""
def longitud_media(distr,d):
    lm = 0
    for i in range(len(d)):
        lm += len(d[distr.at[i,'states']])*distr.at[i,'probab']
    return lm

"""
Dado un dataframe 
devuelve la entropía
"""
def entropia(distr):
    h = 0
    for p in distr['probab']:
        h -= p*math.log(p,2)
    return h

"""
Dada una palabra y un diccionario
devuelve su codificación en binario en el idioma del diccionario
"""
def codifica(palabra, d):
    binario = ""
    for c in palabra:
        binario += d[c]
    return binario

"""
Dada una palabra en binario y un diccionario
devuelve su decodificación en el idioma del diccionario
"""
def decodifica(binario, d):
    palabra = ""
    codigo = ''
    # Se separan keys y values en diferentes listas para poder buscar key por value
    list_values = list(d.values())
    list_keys = list(d.keys())
    for bit in binario:
        # Se buscan los tramos mínimos del binario que constituyen un codigo asociado a un caracter
        codigo += bit
        if codigo in d.values():
            palabra += list_keys[list_values.index(codigo)]
            codigo = ''
    return palabra



# FORMATO
class Formato:
    BOLD = "\033[1m"
    RESET = "\033[0m"

#### Vamos al directorio de trabajo####
os.getcwd()
#os.chdir(ubica)
#files = os.listdir(ruta)

with open('GCOM2022_pract2_auxiliar_eng.txt', 'r',encoding="utf8") as file:
      en = file.read()
     
with open('GCOM2022_pract2_auxiliar_esp.txt', 'r',encoding="utf8") as file:
      es = file.read()
    
# APARTADO i)
print("\n" + Formato.BOLD + "Apartado i)" + Formato.RESET)

# eng
print("\n" + Formato.BOLD + "S_eng:" + Formato.RESET)
tab_en = Counter(en)

##### Transformamos en formato array de los carácteres (states) y su frecuencia
##### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))

tree_en = huffman_tree(distr_en)

# Código de Huffman binario
d_en = extraer_cadena_caracter(tree_en)
print("Código de Huffman binario: " + str(d_en))
# Longitud media
lm_en = longitud_media(distr_en, d_en)
print("Longitud media: " + str(lm_en))
# Entropía
e_en = entropia(distr_en)
# Teorema de Shannon
print("Teorema de Shannon: " + str(e_en) + " <= " + str(lm_en) + " < " + str(e_en+1))

#esp
print("\n" + Formato.BOLD + "S_esp:" + Formato.RESET)
tab_es = Counter(es)

##### Transformamos en formato array de los carácteres (states) y su frecuencia
##### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_es_states = np.array(list(tab_es))
tab_es_weights = np.array(list(tab_es.values()))
tab_es_probab = tab_es_weights/float(np.sum(tab_es_weights))
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab })
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index=np.arange(0,len(tab_es_states))

tree_es = huffman_tree(distr_es)

# Código de Huffman binario
d_es = extraer_cadena_caracter(tree_es)
print("Código de Huffman binario: " + str(d_es))
# Longitud media
lm_es = longitud_media(distr_es, d_es)
print("Longitud media: " + str(lm_es))
# Entropía
e_es = entropia(distr_es)
# Teorema de Shannon
print("Teorema de Shannon: " + str(e_es) + " <= " + str(lm_es) + " < " + str(e_es+1))

# APARTADO ii)
print("\n" + Formato.BOLD + "Apartado ii)" + Formato.RESET)

palabra = 'medieval'
codifica_huffman_en = codifica(palabra, d_en)
codifica_huffman_es = codifica(palabra, d_es)
codifica_usual = ''.join(format(c, 'b') for c in bytearray(palabra, "utf-8"))

print("Codificacion de \"" + palabra + "\" en inglés: " + codifica_huffman_en)
print("   Eficiencia del " + str(len(codifica_usual)/len(codifica_huffman_en)*100) + "%")
print("Codificacion de \"" + palabra + "\" en español: " + codifica_huffman_es)
print("   Eficiencia del " + str(len(codifica_usual)/len(codifica_huffman_es)*100) + "%")

# APARTADO iii)
print("\n" + Formato.BOLD + "Apartado iii)" + Formato.RESET)

binario = '10111101101110110111011111'
print("La palabra '" + binario + "' decodificada del inglés es '" + decodifica(binario,d_en) + '\'')

