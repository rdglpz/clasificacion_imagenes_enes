# Clasificación de Imágenes por objetos y redes neuronales

**3er Examen Parcial de la Unidad 5 y primera Parte de la unidad 7. **

**Profesor:  Rodrigo López Farías**



Nombre:



1. (10%) Describe con tus propias palabras, y si crees necesario, con ayuda de un dibujo, como calcula el árbol de clasificación la estimación de la clase perteneciente a un vector de entrada, recordando que este arbol tiene una profundidad $l>0$ y en sus nodos terminales u hojas tiene asociado un conjunto de datos de número impar mayor a 1. R=
2. (10%) ¿Que ventajas tiene un árbol de clasificación sobre una regresión logística?
3. (10%) ¿Qué mide el índice Gini?.
4. (20%) Calcula el índice Gini de los siguientes conjunto de datos

$Y_1 = \{1,1,4,0\}. G_1 =  \\ Y_2= \{1,1,1,0\}. G_2 = \\ Y_3= \{1,1,0,0\}. G_3 =$ 

5. (20%) Construye y dibuja un árbol de clasificación de profunidad 1 con el algoritmo *greedy* (glotón) visto en clase utilizando la base de datos proporcionada y el índice Gini para evaluar la calidad de las divisiones y contesta las siguientes preguntas. 
   1. ¿Cuál es la variable que proporciona mas información para minimizar los errores de clasificación?.
   2. ¿Cuál es el índice Gini de cada conjunto de datos asociado a cada nodo terminal u hoja.

Ayuda: 

$\hat{\pi}_c = \frac{1}{|N|}\sum_{i\in N} I (y_i=c)$

$\sum_{c=1}^C \hat{\pi}_c(1-\hat{\pi}_c) = 1 - \sum_{c}\hat{\pi}^2_c$

$c\in\{0,1\}$

```table
|entrada|salida|
|x_1|x_2|   y  |
|---|---|------|
| 1 | 0 |   1  |
| 1 | 0 |   1  |
| 1 | 0 |   0  |
| 1 | 1 |   0  |
| 0 | 1 |   0  |
| 1 | 1 |   0  |
```



6. (30%) Dado un modelo de neurona simple (perceptrón) definido por las siguientes ecuaciones, obtener las derivadas parciales $\frac{\partial E}{\partial w_1},\frac{\partial E}{\partial w_2}$ y  $\frac{\partial E}{\partial b}$.

<img src="/Users/rodrigo/Downloads/mtheme_cgeo/figures/simple_classifier_structure.png" alt="simple_classifier_structure" style="zoom:50%;" />





Dónde:

* Cálculo del error  $E(z_i,\hat{z}_i)$ de la respuesta de salida $\hat{z}_i$es la log verosimilitud.

  E= $(z \text{ log } \hat{z} + (1-z) \text{ log } [1-\hat{z}])$

Pista:

$\frac{\partial E}{\partial \hat{z}} =  \frac{\hat{z}-z}{(\hat{z}-1)\hat{z}}$

* La función de activación es la misma respuesta de salida que está dada por la función sigmoide 

  $\hat{z} = \frac{1}{1+e^{-y}}$

$\frac{\partial \hat{z}}{\partial y} = ? $

* El valor de $y$ esta dado por: 

  $y= w_1x_1+w_2x_2+b$

  $\frac{\partial {y}}{\partial w_1} = ? $

  $\frac{\partial {y}}{\partial w_2} = ? $

  $\frac{\partial {y}}{\partial b} = ? $

  
  
  donde:
  
   $\{x_1,x_2\}$ son los valores de entrada, 
  
  $w_1,w_2,b$ son los parámetros optimizables de la función. 



Calcular las derivadas parciales con respecto a $w_1,w_2,b$

$\frac{\partial E}{\partial w_1}=$

 $\frac{\partial E}{\partial b}=$

