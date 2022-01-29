# Práctica 1: ATRACTOR LOGÍSTICO

### Tutorial introducción Python
#### Diferentes formas de crear variables vacías (array y listas)
```python
n1 = np.empty([10])
n2 = [0]*10
n3 = [0 for x in range(10)]
print(n1) # [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
print(n2) # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(n3) # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(type(n1)) # <type 'numpy.ndarray'>
print(type(n2)) # <type 'list'>
print(type(n3)) # <type 'list'>
```

#### Ojo con la multiplicación de listas
```python
# [-1,-1]*[1,2]
a = [-1,-1]
b = [1,2]
print(np.array(a)*np.array(b)) # [-1,-2]
print(np.multiply(a,b)) # [-1,-2]
print([a0*b0 for a0,b0 in zip(a,b)]) # [-1,-2]
print([a[i]*b[i] for i in range(len(a))]) # [-1,-2]
print(map(lambda x,y: x*y, a,b)) # [-1,-2]
```

## Indicaciones
- Realizar 200 ó 100 iteraciones (no 3).

#### Diferencia entre picos (no entre picos y valles)
Observamos las diferencias: 0,5 - 0,3 - ...
En la iteración 4 tiene error de 0,2.
Si yo sé el signo puede ocurrir que en algunos casos vaya oscilando.
Si yo sé el signo puedo conocer que ese valor será como mucho tan alto como el anterior.
El valor del actual valle (x) es menor que el valor del anterior (4,43) (en 0,1 p.e.).
Cuanto más avances en el tiempo menor será esta diferencia.
Si mi estimación de error es 0,1 el '3' del 4,43 me sobra.
Si el error es de 0,2 se quita el 3.
Si el error es de 0,10 - 0,19 se puede mantener el 3.
4,4322 +- 0,04 quito el 22.
(Solo me equivoco en el '7') Notación: 4,43235227(5), también sirve +- 5·10^-8.

*Si no se especifica, la última cifra significativa me está dando el error, p.e. 3,51 +- 0,01*

## Desarrollo
#### Cómo entender la práctica
Cada conjunto V0 contiene los estados que adopta cierta orbita cuando tiende a infinito.
Es decir, tras cierto punto inicial x0 y en cierta etapa r, se estudia la evolución de la órbita y los estados a los que converge para luego representarlos como puntos y formar el árbol de Feigenbaum.

## Dudas
#### Cálculo de error
Si tenemos la siguiente órbita
[0.58292956, 0.74389456, 0.58277908, 0.74378548, 0.58238252, 0.74329452]
que podemos separar en los siguientes ciclos
V0 = [0.58292956, 0.74389456]
V1 = [0.58277908, 0.74378548]
V2 = [0.58238252, 0.74329452]
¿el error se calcula de manera relativa, respecto al inmediatamente anterior, o absoluta respecto al primero?

## Preguntas típicas de examen
- Qué es una orbita, qué es una atracción, etc.
