# Tarea 9 

**Redes neuronales y clasificación de  contexto**



Utilizando redes neuronales de propagación hacia adelante multicapa (perceptrones multicapa):

Realizar clasificación de contexto con alguna imagen satelital con el fin de detectar objetos considerando su contexto.

A partir de una imagen, programar una ventana deslizante de tamaño $n \times n$, que extraiga subsecciones o recortes de la imagen. 

Seleccionar una muestra de todos los recortes y clasificarlas manualmente. (por ejemplo, si el objetivo es clasificar lagos, clasificar como lago los recortes de la imagen que contienen lagos).



a) Crear una imágen binaria clasificada.

b) Crear una imagen de probabilidades.

c) Hacer validación cruzada. Del conjunto de datos seleccionados sugerido 70% entrenamiento 30% prueba.

d) Cuidar balanceo de clases.

Notas:

La ventana debe ser de un tamaño suficiente para poder identificar un objeto por su contexto.

* Recomiendo que: 

1. Seleccionen un problema donde los objetos tengan aproximadamente el mismo tamaño. 
2. Descarten en la medida de lo posible aquellos recortes con solo una parte del objeto. (Pueden determinar un umbral, por ejemplo, si el recorte tiene mas de $m$ pixeles, tomarlo en cuenta como recorte con un objeto de interés dentro.)

* Una vez entrenada la red neuronal, volver a recorrer la ventana deslizante por toda la imagen, y por cada posición de la ventana generar un vector unidimensional de tamaño $X_{(n \cdot n, 1)}$ a partir de un recorte bidimensional  $X_{(n,n)}$que será el vector de entrada a la red neuronal. 



**Ponderación:**

1) 20% Definir el problema

2) 20% Generar de manera adecuada el conjunto de datos. El conjunto de datos de entrenamiento y prueba y de cada clase deben ser representativos. Sugerencia, modificar el programa para generar recortes.

3) 20% Reportar la precisión del clasificador utilizando la matriz de confusión.

3) 20% Generar la imagen clasificada.

4) 20% Generar la imagen de probabilidades de clasificación.
