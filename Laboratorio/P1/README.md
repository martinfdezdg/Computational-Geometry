# Práctica 1: ATRACTOR LOGÍSTICO

## Introducción Python
### Diferentes formas de crear variables vacías (array y listas)
```python
n1 = np.empty([10])
n2 = [0]*10
n3 = [0 for x in range(10)]
print(n1)
print(n2)
print(n3)
print(type(n1))
print(type(n2))
print(type(n3))
```

## Número de iteraciones
Realizar 200 o 100 (no 3).

## Diferencia entre picos (no entre picos y valles)
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



**Preguntas típicas**
Qué es una orbita, qué es una atracción, etc.
