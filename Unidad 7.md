# Unidad 7 Clasificación de Imágenes. Técnicas Avanzadas.

## Clasificación por redes neuronales

* Una red neuronal Artificial es un modelo matemático inspirado en el comportamiento biológico de las neuronas y en la estructura del cerebro. 
* Las ideas y descubrimientos en neurociencias sirven como inspiración en la construcción de modelos matemáticos - computacionales capaces de resolver problemas complejos de clasificación y regresión como son las redes neuronales artificiales. 
* Un esquema simplificado de una neurona biológica de un tipo se muestra en la siguiente figura.

![Screen Shot 2021-08-19 at 18.30.46](figures/dendrita.png)

* La figura muestra las partes de una neurona que son de interés para la construcción de estos modelos matemáticos.
  * SOMA: Cuerpo central de la neurona la cual procesa la información de entrada y la transfiere al axón.
  * Axón: es la prolongación del SOMA . 
  * Dendritas: reciben información que transfieren al SOMA.
  * Sinapsis. Zona de conexión entre una neurona y otra.

La neurona transmite los impulsos nerviosos a otras neuronas interconectadas. Estos viajan desde las dendritas hasta el axón. El axón se comunica con otras neuronas por sinapsis (comunicacación asoma-dendrita) construyendo de esta manera una red de comunicación neuronal.

 La sinapsis constituye el sitio físico que sirve de puente para el paso de información de una neurona a otra, permitiendo que las diferentes partes del sistema interactúen funcionalmente.

Este modelo simplificado ha servido de inspiración para la construcción de estos modelos.

El primer modelo neuronal llamado modelo McCulloch-Pitts surgió en 1943 fué desarrollado por: 

* El psiquiatra y neuroanatomista Warren McCulloch y 
* El Matemático Walter Pitts.

Se apoyó de las matemáticas para simular y explicar el comportamiento de la neurona. 

Este modelo neuronal es bastante simple. 

Es una combinación lineal de un vector entrada de dos dimensiónes al cual se le suma una constante (bias) , y regresa una salida.



El experimento demuestra una analogía donde:

* El vector de entrada $\mathbf{x}^T = [x_1,x_2]$ representa un estímulo de un entorno externo que recibe la neurona artificial.

* La salida $z$ es la respuesta del estímulo. 

* La salida $z$ se va adaptando a su entorno al ajustar los pesos sinápticos $W^T = [w_1,w_2]$ y el termino aditivo o "bias".

En matemáticas son los parámetros optimizables de un modelo de regresión.

Matemáticamente el modelo se describe como 

$$z =  f (w_1x+w_2y+b) = f (\mathbf{W}^T \mathbf{x}+b)$$ (X es vector columna y W es un vector fila)



$f$ es la función de activación que simula el procesamiento que se produce en el soma. 

Así nace el *perceptrón*! que es técnicamente lo mismo que la regresión logística (lineal) con la diferencia que este no se limita a solo dos entradas impuesta por el modelo de McCulloch-Pitts 



### Perceptrón Multicapa



La base del perceptrón multicapa son los perceptrones simples.

Estos perceptrones simples estan interconectados entre si dando lugar a diferentes tipos de arquitecturas en redes neuronales como:



![neuralnetworks](figures/neuralnetworks.png)

Perceptrón multi capa. Esta compuesto por perceptrones organizados por capas que propagan el estímulo  hacia adelante a través de ellas.

Redes neuronales recurrentes.





Un perceptrón Multicapa con propagación hacia adelante tiene la característica de  tener sus neuronas organizadas por capas y cada capa se limita a transferir la información hacia adelante desde la capa de entrada hasta la capa de salida.



![DFFN](figures/DFFN.png)

* Una red neuronal de propagación hacia adelante es una función universal de aproximación con 3 tipos de capas:
  * Capa de entrada
  * Capas intermedias 
  * Capa de Salida



El entrenamiento de este tipo de funciones al igual que los modelos lineales de regresión y clasificación esta compuesto por:



1. Evaluación
2. Corrección

* Evaluación:

  * Evaluación del modelo dado un conjunto de parámetros $\theta$ utilizando una función de costo $C(\theta)$

* Corrección:

  * Calculo de la Derivada del modelo.

    * Ejecutar el algoritmo adecuado. En redes neuronales se utiliza un algoritmo llamado backpropagation.

    

    $\hat{y}= f(g(x,\theta_g),\theta_x)$

    $\frac{{\partial \hat{y} }}{{\partial {w \in {\theta_g \theta_x} }}}$

    * Actualización de los parametros $\theta$. En redes neuronales esto se hace con un algoritmo que propaga el error de la última neurona hacia atras.



Evaluar una función que representa una red neuronal tiene sus consideraciones.





## Regla de la cadena del cálculo

La regla de la cadena del cálculo es utilizada para calcular derivadas de funciones compuestas (e.g., $y = f(g(x))$) por otras funciones de las cuales se conocen sus derivadas.

La regla de la cadena es fácil de escribirla algebraicamente. 



Ejemplo:

Sea $x$ un número real, y $f$ y $g$ funciones de mapeo. Supongamos que tenemos las funciones anidadas ($g o f$) $z=f(g(x))$ y queremos calcular su derivada parcial por la regla de la cadena.

 

$$z = f(g(x))$$

$y=g(x)$

$$z = f(y)$$

 La regla de la cadena nos dice que para calcular la derivda de $z$ podemos hacerlo de esta manera:

 $\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx} $

Esto se puede generalizar para calcular funciones que reciben vectores. 

Supongamos que $\mathbf{x}\in {R}^m$ , $\mathbf{y}\in R^n$, $g$ mapea de $R^m$ a $R^n$, y $f$ mapea de $R^n$ a $R$. Si $\mathbf{y}= g(\mathbf{x})$ y $z = f(\mathbf{y})$

$$\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$



El algoritmo propagación hacia atrás utiliza el principio de la regla de la cadena. De tal manera que el cálculo de la derivadada parcial de redes neuronales con muchas neuronas (y por lo tanto parámetros), sea eficiente guardando parte del cálculo de la derivada númerica en los nodos.



**Ejemplo Formulación de backpropagation con una neurona para resolver un problema de regresión lineal **



La neurona simple tiene la siguiente forma (igual que la regresión lineal)

<img src="/Users/rodrigo/Downloads/mtheme_cgeo/figures/simple_classifier_structure.png" alt="simple_classifier_structure" style="zoom:50%;" />



Las operaciones que se realizan (de atras hacia adelante) son

Cálculo del error $E(z_i,\hat{z}_i)$

$E= \frac{1}{2} \sum_{i\in \text{training}}^n (z_i-\hat{z}_i)^2$

Derivada del Error 

$\frac{\partial E}{\partial \hat{z_i}} = - \sum_{i}^n (z_i-\hat{z}_i)$



Cálculo de la salida $\hat{z}_i$

$\hat{z}_i = \text{sigm}(y) = \frac{1}{1+e^{-y}}$

Calculo de la derivada de  $\hat{z}_i$

$ \frac{\partial \hat{z}_i}{\partial y} =  \hat{z}_i(1-\hat{z}_i)$



Cálculo de $y$ 

$y = \sum x_i w_i$

Derivadas.

$ \frac{\partial y}{\partial w_i} =  x_i$

$ \frac{\partial y}{\partial x_i} =  w_i$



Para calcular la derivada del error con respecto a los parámetros $\mathbf{w}$, 

$\frac{\partial E}{\partial w_i} = \frac{\partial y}{\partial w_i} \frac{\partial \hat{z}_i}{\partial y} \frac{\partial E}{\partial \hat{z_i}}$ 

$\frac{\partial E}{\partial w_i} =  -  \sum_{i}^n x_i \hat{z}_i(1-\hat{z}_i) (z_i-\hat{z}_i)$ 






