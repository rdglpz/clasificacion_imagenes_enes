# Clasificación de Imágenes por objetos y redes neuronales

Dr. Rodrigo López Farías



## Unidad 1. Introducción a las técnicas de Clasificación

1.0 Introducción al  aprendizaje automático.

* En los primeros días de la inteligencia artificial, este campo tomó y resolvió los primeros problemas que intelectualmente son dificiles para los humanos pero muy fáciles para las computadoras. 
* Se ha cuestionado que es inteligencia Artificial. Si no es inteligencia humana que es?  
  * Deep y deeper Blue que derrotaron al ajedrecista Gary Kaspárov en 1996. Es una Inteligencia Artificial? ![450_1000](	figures/450_1000.jpg)

https://www.professional-ai.com/deep-blue-algorithm.html



Retos de la inteligencia Artificial

* Resolver problemas fáciles e intuitivos  pero que son difíciles de describir formalmente para las personas. 
* Por lo tanto parte de este problema consiste en resolver como el conocimiento informal debe ser representado en una computadora.

![imagematrix](figures/imagematrix.png)

Representación de una Imagen en una Matriz.  

Ejemplos de problemas que son fáciles para los humanos pero dificiles para las computadoras.

* Entender el lenguaje natural.
* Inferir el sentimiento de las personas.
* Traducción Automática
* Clasificación de Imágenes.
* Reconocimiento de objetos.
* Qué otros problemas los humanos pueden resolver fácilmente? 
  * La Justicia



Hay otros problemas que van mas allá de tema pero vale la pena mencionar son, aquellos problemas de decisión dificiles para las computadoras y para los humanos. El problema es si los humanos delegamos a una computadora tomar decisiones éticas.  

* Decisiones éticas. 

<img src="figures/m.i.tsmartcarchoicescolo01-9.jpg" alt="m.i.tsmartcarchoicescolo01-9" style="zoom:24%;" />

https://www.technologyreview.com/2018/10/24/139313/a-global-ethics-study-aims-to-help-ai-solve-the-self-driving-trolley-problem/



Por lo tanto, el Aprendizaje automático consiste en dar solución a estos problemas intuitivos aprendiendo de la experiencia. 

En aprendizaje profundo, la idea principal es  entender la información como una jerarquía de conceptos complicados a partir de otros conceptos mas simples.

En los inicios se intentó crear inteligencia artificial con codificación "dura". (Introducción explicita de reglas de inferencia en una base de datos llamado Enfoque de **Conocimiento Base** para la inteligencia artificial)

La rigidez de este tipo de enfoque, sugiere que los sistemas de IA, deben tener la habilidad de adquirir su propio conocimiento, extrayendo patrones de los datos en bruto. Esta capacidad de las computadoras se le llama Aprendizaje Automático (AA) o **"Machine Learning"**.

AA pudo hacer que las computadoras aborden problemas complejos para los humanos utilizando el conocimiento del mundo real  para tomar decisiones que parecen subjetivas. Por ejemplo, uno de los primeros éxitos es la implementación de una regresión logística simple que es capaz de recomendar o no parto por cesarea a los médicos.  (Principio utilizado por las redes neuronales) 

![curva-sigmoide](figures/curva-sigmoide.png)

El truco para que funcionen estos modelos reside en **una buena representación de los datos** que incluye una **selección de característicaís** cuidadosa  por un experto.

Una mala selección o representación de datos, genera malos algoritmos predictivos. 

Para muchas tareas del mundo real es muy complicado distinguir las características que deben ser utilizadas para entrenar un modelo. 

Por ejemplo, como distinguir una persona de un carro, o de un perro.

Para esto, una solución es utilizar **AA para aprender automáticamente la representación misma de los datos**. 

Cuando se diseñan algoritmos para aprender características, una de las metas es distinguir los factores invariantes o características de los factores de variación  o ruido que explican los datos observados. 

<img src="figures/download-2.jpg" alt="download-2" style="zoom:150%;" />



![Screen Shot 2021-02-16 at 21.14.53](figures/Screen Shot 2021-02-16 at 21.14.53.png)

ML responde a la pregunta de cual es el mejor modelo para explicar datos.

Ejemplo de como el aprendizaje profundo extrae características importantes

Que características invariantes representan a un gato o a un perro?

**Machine Learning es una subárea de de la IA que detecta patrones de manera autoática en los datos**



### Antecedentes

****

Los modelos predecesores, están basados en modelos lineales simples, que trataron de asociar (o mapear) una entrada de tamaño $n$ $ \mathbf{x}= \{x_1,x_2,\dots,x_i,\dots,x_n\}$ a una salida $y_i$. 

Estos modelos aprenden un conjunto de pesos $\mathbf{w} = \{w_1,w_2,\dots,w_n\}$., para ajustar una funcion $f$:

$$f(\mathbf{x},\mathbf{w}) = x_1 w_1+x_2 w_2 + \dots+x_n w_n$$ 

tal que la suma de los errores $\sum ( f(\mathbf{x}_i, \mathbf{w}) - y_i)$ sea la mínimo.



**Ecuación de la recta**



$$y_i = \mathbf{w}x_i + b $$



**Primera Tarea: 1 ejemplo de aplicación de la ecuación de la recta.**



Ola Cero

 (McCulloch and Pitts, 1943) Utiliza el modelo lineal simple 

$$ sign(f(\mathbf{x},\mathbf{w})) \in \{-1,1\}$$ 

Este modelo puede reconocer dos categorias. Los pesos $\mathbf{w}$ son ajustados manualmente. **No existe un mecanismo de entrenamiento.**

En ese tiempo sale la primera computadora digital de propósito general: ENIAC 

<img src="figures/9DN1KtJrjR_1415655346589.jpg" alt="9DN1KtJrjR_1415655346589" style="zoom:50%;" />



Primera ola: 

Cibernética (1940-1960, Rosenblatt 1958). Impulsado por la teoría de aprendizaje Biológico. Nace el Modelo Perceptrón., que es el primer modelo de una sola neurona capaz de aprender de los datos.

![0_wOYoifz24Wz_I152](figures/0_wOYoifz24Wz_I152.png)

https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b



(Minsky and Papert, 1969) Estos modelos lineales tienen limitaciones. El mas famoso es la incapacidad de resolver la función XOR. 

![0_qdRb80zUpJPtrbRD](figures/0_qdRb80zUpJPtrbRD.gif)



**Tabla de verdad de la operación lógica xor**

x | xor | y 

0 | xor |0 = 0

0 | xor | 1 = 1

1 | xor|1 = 0

1 |xor | 0 = 1 

Segunda ola: Conexionismo: (1980-1995, Rumelhart ) Impulsado por por red neuronal con una o dos capas ocultas que resuelve el problema del XOR agregando una capa oculta.

<img src="figures/0_158hcRQzzw_wpEZW.png" alt="0_158hcRQzzw_wpEZW" style="zoom:50%;" />



Tercera ola: 1995 - Deep learning. Impulsado por Redes Neuronales Multi-Capa y algoritmo De Gradiente Estocástico Descendente.



<img src="figures/DFFN.png" alt="DFFN" style="zoom:50%;" />



<img src="figures/redesTopologia.jpg" alt="redesTopologia" style="zoom:50%;" />

Hochreiter (1991) y Bengio et al. (1994), Se identifico otro problema con el aprendizaje profundo que eses el modelado de secuencias muy largas, para esto Hochreiter y Schmidhuber (1997) propusieron  the long short-term memory (LSTM) para resolver estos problemas. LTSM es una red neuronal recurrente que incorpora una operación para definir el que tanto importa recordar la información para hacer una Clasificacióno regresión.

![The_LSTM_Cell.svg](figures/The_LSTM_Cell.svg.png)





<img src="figures/venn diagram.png" alt="venn diagram" style="zoom:20%;" />

Deep Learning son un tipo de aprendizaje automático que, ademas de encontrar una función de mapeo para asociarl un conjunto de características $\mathbf{x}$ con una salida datos ($\mathbf{y}$), internamente logran aprender como representar esos datos.  



<img src="figures/deeplearning.png" alt="deeplearning" style="zoom:25%;" />



Técnicas modernas de AP, toman ideas de muchas otras áreas, como 

* Álgebra Lineal
* Probabilidad
* Teoría de la información
* Optimización Numérica



## Tipos de Aprendizaje automático

Tomando como guía la figura del diagrama de Venn,  y enfocándonos en el área de Aprendizaje Automático, podemos identificar  dos grandes  de tipos de aprendizaje. 

###  1) Aprendizaje Supervisado

El aprendizaje supervisado trata de aprender una función de mapeo aproximada $\hat{f}: \mathbf{x}_i \rightarrow {\hat{y}_i} $ a la función real $f:\mathbf{x}_i \rightarrow {y_i}$ dado  a un conjunto de pares de datos etiquetados

 $$D = \{(\mathbf{x}_i,y_i)\}^N_{i=1}$$ 

donde:

* $D$: es el conjunto de entrenamiento. 

* $N$: número de muestras

* $\mathbf{x}_i$: Atributos o características, pueden ser discretas, continuas o reales.

* $y_i$: Variable Categórica con valores **(Clasificación)** $y_i \in \{1,\dots,C\}$, o **(Regresión)** $y_i \in R^{Dim}$

* Si $C = 2$ significa que trata de un problema de **clasificación binaria**.

* Cuando C>2, se le llama **clasificación Multi-clase.**

* Si un objeto puede pertenecer a diferentes clases al mismo tiempo se le llama **clasificación Multi-etiqueta**. 

* La meta principal es ajustar una función de aproximación con capacidad de **generalización**  para clasificar de manera adecuada los datos mas allá del conjunto de entrenamiento.

  

  

  ![Screen Shot 2021-02-23 at 20.40.29](figures/Screen Shot 2021-02-23 at 20.40.29.png)

  Ejemplo:

* Árboles de clasificación
* Clasificación por mínima distancia
* Redes neuronales. 
* Modelos gausianos
* Modelos Bayesianos
* Maquina de Soporte de Vectores

### 2) Aprendizaje No supervisado

Este tipo de aprendizaje tiene un enfoque descriptivo con datos que no están previamente etiquetados. El conjunto de datos se se representa como 

$$D = \{\mathbf{x}_i\}^N_{i=1}$$.

 La finalidad es encontrar la estructura general de los datos. Es de útilidad para 

* Utilizado para etiquetado automático, cuando el etiquetado supervisado es imposible o muy costoso.
* Resuelve el problema de encontrar clúster o grupos de datos en los datos.
* Descubre factores latentes. (aquellas variables que dan mas información sobre el conjunto de datos).
* Estimación de densidades de distribución.

Aplicación de ejemplo: Descubrir cuantos grupos de datos existen en los datos. Lado Izquierdo, datos sin clasificar del peso y la altura de cada persona. En la derecha detección de dos grupos diferentes.

* <img src="figures/Screen Shot 2021-02-23 at 16.14.24.png" alt="Screen Shot 2021-02-23 at 16.14.24" style="zoom:50%;" />

Modelos relacionados

* k-medias
* Mapas auto-organizados
* Dendogramas
* 

Otros modelos no supervisados se pueden consultar en 

https://www.cienciadedatos.net/documentos/37_clustering_y_heatmaps#Ejemplo_divisivo

### Predicción Probabilista en la clasificación de los datos.

* La predicción probabilista nos permite manejar casos que no son claros para el humano. 

* Es deseable que exista una función de probabilidad la cual para una entrada, arroje una probabilidad de pertenecer a una clase. 

* La probabilidad de que un  dato o vector de entrada pertenezca a una clase se puede expresar utilizando **probabilidad condicional**:

  $$p(y|\mathbf{x},D)$$


  Donde $D$: es el conjunto de datos, $\mathbf{x}$ es el vector de entrada y la función regresa un valor entre 0 y 1.

  Para seleccionar la mejor estimación se utiliza un estimador de **Maximum a posteriori**

  $$\hat{y} = \hat{f}(\mathbf{x}) = \text{ argmax }_{c}  p(y=c|\mathbf{x},D)$$

donde $c\in \{1,\dots,C\}$

### Aplicaciones

* Clasificación de Documentos y filtro de spam. (La probabilidad de que un documento pertenezca a spam)
* Clasificador de flores.
* Clasificación de imágenes 
* Reconocimiento de escritura
* Encontrar objetos dentro de una imagen con ventana deslizante (detección y localización de rostros).
* <img src="figures/Screen Shot 2021-02-23 at 20.41.12.png" alt="Screen Shot 2021-02-23 at 20.41.12" />
* Predicción de series de tiempo
* Predecir la edad de un espectador dado un video de YouTube.
* Predecir la temperatura dentro de una locación, utilizando datos atmosféricos, hora del día, materiales de construcción, sensores  etc...
* Predicción de concentraciones de contaminantes.

### Descubrimiento de agrupamientos:
El descubrimiento de agrupamiento (o clustering tiene dos objetivos)
  * Objetivo 1.  Estimar la distribución de probabilidad de un conjunto de datos $D$ con $K$ grupos. Esto se puede expresar como la probabilidad de tener $K$ grupos dado un conjunto de datos $D$

$$p(K|D)$$

Para encontrar $K$, se requiere una manera de medir el ajuste de los datos los diferentes $K$ grupos, hasta maximizar la probabilidad de que el conjunto de datos $D$ sea generado por una distribucion de probabilidad.

$K^* = argmax_K p(K|D)$

Objetivo 2:  Estimar a que grupo pertenece un dato. 

Por ejemplo haciendo $z_i \in\{1,\dots,K\}$, donde cada punto o dato $i$, es asignado a una de esas clases. 

Como $z_i$ es una variable calculada y no existe en los datos de entrenamiento, a esta se le llama **Variable Latente**, o variable oculta.

Se puede estimar o inferir a que clase pertenece calculando: 

$$z_i^*=argmax_k p(z_i=k|\mathbf{x}_i, D)$$,

Que significa, calcular la clase $k$ que que maximiza la función de probabilidad dado un dato $\mathbf{x}_i$ en el conjunto de datos $D$

Ejemplo:

![Screen Shot 2021-02-23 at 20.41.29](figures/Screen Shot 2021-02-23 at 20.41.29.png)

### Descubriendo variables (o factores latentes)


* Útil cuando los datos son de dimensiones muy altas.

* Es útil para conocer el subconjunto de variables de un vector de características que conservan la máxima información. (Podemos reconstruir la información a partir de información *clave*)

* Se proyectan en un espacio de dimesión reducida $M: R^m \rightarrow R^{n}$

* Por lo regular  2 <= n <=3 para poder analizar los datos visualmente.

* Util para compresión de imágenes o aplicar algoritmos de agrupamiento.

* Uno de los métodos mas útilizados es análisis de componentes principales o PCA.

* ![Screen Shot 2021-02-23 at 20.41.44](figures/Screen Shot 2021-02-23 at 20.41.44.png)

  

  


## Explicaciones conceptuales 

### Modelos paramétricos y no paramétricos

**Modelos paramétricos**. 

* Son modelos que tienen un número fijo de parámetros independientemente de la cantidad de datos. 
* Requiere un conocimiento sobre el fenómeno mas detallado.
* Se apoya de supociciones fuertes sobre los datos. (e.g. Datos distribuidos con cierta función de probabilidad)
* Funciona bien con pocos datos.
* No sufren de **sobre-ajuste**.

Ejemplos  

1. Regresión lineal.
2. Estimación de distribuciones de probabilidad.    

 **Modelos No-paramétricos**: 

* Los parámetros crecen con los datos.

* Se requiere un conocimiento menos profundo sobre el fenómeno.

* No requiere hacer supocisiones fuertes sobre los datos. 

* Requiere muchos datos para tener buen desempeño

* Sufren de sobre-ajuste 

  

  **Ejemplos:**

  Redes neuronales (Supervisado)
  
  Redes neuronales convolucionales (Supervisado)
  
  $k$-vecinos cercanos (Supervisado)
  
  PCA (no-supervisado)
  
  Mapas auto-organizados (No paramétricos - No supervisados)
  
  



Ademas de los modelos de clasifiación mencionados existen otros modelos clasificadores que también son de importancia en este curso

* Minima distancia
* Modelo del Paralelepípedo
* Árboles de clasificación
* Clasificación por máxima verosimilitud

### Ejemplos de Modelos paramétricos para clasificación.

**Clasificador Lineal**

Un clasificador lineal es una función de la forma 

$$y = f( \mathbf{w}^T\mathbf{x} + b)$$

donde , $\mathbf{w}$ es un vector de pesos, $\mathbf{x}$ es el vector de entrada y la función $f$ convierte el producto interno de los dos vectores en una salida. 

la función $f$ toma valor de:

* 1: si $\mathbf{w}^T\mathbf{x}+b > 0$, y
*  $0$: de otra manera.

Para el problema de clasificación de dos clases, el problema puede interpretarse como un hiperplano que divide el espcio de vectores de entrada $x$, en dos. Donde los valores que quedan de un lado del hiperplano son clasificados como positivos y lo que quedan debajo negativos

![440px-Svm_separating_hyperplanes](figures/440px-Svm_separating_hyperplanes.png)



**Regresión Logística**

Es una generalización de la regresión lineal para clasificación binaria.

La respuesta o salida $y$ de la función responde a una distribución de probabilidad de Bernoulli. 

$$p(y|\mathbf{x,w}) = Ber (y| \mu(x)$$)

 donde la respuesta toma dos valores: $y\in\{0,1\}$. y $\mu(\mathbf{x}) = p(y=1|x)$  es la probabilidad de exito dada una entrada $\mathbf{x}$.

La función de distribución de probabilidad está dada por la función sigmoide:

$$\mu(\mathbf{x})=\frac{1}{1+exp(\eta)}$$ 

Acoplando  los términos en orden obtenemos la **regresión logística**

$$p(y|\mathbf{x,w}) = Ber(y|sigm(\mathbf{w^Tx}))$$

La distribución de probabilidad de Bernoulli es discreta y dicotómica que calcula la probabilidad de éxito. 

Ejemplo:

$$p(y_i=1|x_i,\mathbf{w}) = sigm(w_0+w_1x_i) $$

es aplicado a definir si un estudiante pasa la prueba ($y=1$) o no ($y=0$) de acuerdo a su puntuacion.

Para esto a partir de los datos ($x_i$,$y_i$), donde $x\in R$ es la calificación o puntaje representado por el eje horizontal, y $y\in{0,1}$ (pasa o aprueba la clase), representado por el eje vertical. 

Los puntos negros son los datos reales.

Los rojos son salidas estimadas $p(y=1|\mathbf{x}_i,\hat{\mathbf{w}})$

Si el umbral es mayor a 0.5 entonces la salida estimada será $\hat{y}(x) = p(y=1|\mathbf{x})>0.5 = 1$ 

 ![Screen Shot 2021-02-23 at 21.47.42](/Users/rodrigo/SourceCodes/git/clasificacion_imagenes_enes/figures/Screen Shot 2021-02-23 at 21.47.42.png)





### Ejemplo de modelo no paramétrico

**$k$ vecinos cercanos**

Este algoritmo consiste en buscar aquellos datos cercanos a un *dato de prueba* dado un radio $r$ o tamaño de vecindad $K$. Y regresa la proporcion de clases encontradas. 

Este algoritmo puede expresarse como

$$p(y=c|\mathbf{x},D,K)=\frac{1}{K} \sum_{i\in N_k (\mathbf{x},D)} I(y_i=c)$$

Donde $N_k(\mathbf{x},D)$ son los índices pertenecientes al vencindario.

$I(e)$ es la funcion indicadora que devuelve $1$ si $e$ es verdadero y $0$ en caso contrario.

con $K=1$ se genera un diagrama de Voronoi. 

![Screen Shot 2021-02-23 at 21.01.09](figures/Screen Shot 2021-02-23 at 21.01.09.png)

<img src="figures/Screen Shot 2021-02-23 at 20.42.30.png" alt="Screen Shot 2021-02-23 at 20.42.30" style="zoom:25%;" />

1) Datos sintéticos de 3 clases

2) Probabilidad de cada punto de pertenecer  a clase 1 con k=10 vecinos cercanos.

3) Probabilidad de cada punto de pertenecer  a clase 2 con k=10 vecinos cercanos.

4) Áreas de cada clase construidas con los mismos datos. Estas representan la clase con mas probabilidad de pertenecer a un dato $\mathbf{x}$ . Se puede expresar como: $\hat{y}=argmax_c(y=c|\mathbf{x},D)$



### Maldición de la dimensionalidad.

Se refiere a la incapacidad o el mal desempeño de los modelos de aprendizaje automático de poder escalar con los datos debido a la alta dimensionalidad del problema. 



![Screen Shot 2021-02-23 at 20.42.46](figures/Screen Shot 2021-02-23 at 20.42.46.png)

En el caso de vecinos cercanos, la distancia para obtener una vecindad mínima crece exponencialmente con el número de dimensiones. 

En optimización el espacio de búsqueda crece exponencialmente con el número de dimensiones del espacio.

## Etapas del proceso de clasificación

0. Definir el problema de clasificación (o regresión)
1. Colectar la mayor cantidad de datos posible con la mejor calidad.  Ordenarlos en una tabla ( como .csv)
2. Preparar los datos
   1. Aleatorizar los datos para que el orden entre ellos no afecte en el entrenamiento del modelo.
   2. Hacer un análisis exploratorio de los datos para detectar características generales de las correlaciones, variables de interés, balance de clases, formas generales de las distribuciones.
   3. Reducir la dimensionalidad de ser posible.
   4. Definir el conjunto de entrenamiento y prueba.
3. Entrenar y probar varios modelos bajo las mismas condiciones.
4. **Seleccionar el modelo**
5. En caso de no tener resultados satisfactorios, ajustar parámetros o incluir nuevos modelos.
6. Predicción, clasificación o inferencia.
7. Interpretar el modelo





### **Selección de modelo**

Cuando tenemos una variedad de modelos de diferentes tipos, existe la necesitad de seleccionar el mejor modelo.

Para esto se requiere calcular el error generalizado, el cual es el error esperado de la tasa de clasificaciones incorrectas de los datos futuros.

Para esto es necesario organizar los datos en 1 o mas particiones de conjunto de entrenamiento, y prueba (Fig 1.21). 

Una manera de validar el model es el siguiente:

0. Seleccionar el modelo de clasificación

1.  Particionar los datos en dos conjuntos (k-fold validation): 

   1. Entrenamiento 
   2. Prueba

2. Para una configuración de la partición: entrenar un modelo con el conjunto de entrenamiento y probar con con el conjunto de prueba.

3. Medir el error generalizado con el conjunto de prueba.

4. (opcional) Repetir el procedimiento varias veces seleccionando una configuración diferente.

   

En cada partición, el conjunto de entrenmiento es utilizado para ajustar parametro del modelo, y el conjunto de prueba para calcular el error y evitar seleccionar un modelo **sobre ajustado**.

Este error puede ser aproximado utilizando un conjunto de datos de prueba. 

 Ejemplo de Validación Cruzada con un ejemplo de k vecinos cercanos.

![Screen Shot 2021-02-23 at 20.44.22](figures/Screen Shot 2021-02-23 at 20.44.22.png)

Tarea 3 : Variaciones de validación cruzada

### Sobre ajuste

Es un problema que sufren principalmente los modelos pertenecientes a aprendizaje automático. Un ajuste *perfecto* sobre el conjunto de entrenamiento impide desempeñarse de manera adecuada con cualquier otro dato que no haya sido contemplado en su entrenamiento. Esto se detecta cuando tenemos un error de ajuste muy bajo en el conjunto de prueba y un error muy alto en el conjunto de entrenamiento.





### No Free Lunch Theorem



Se refiere a la no existencia de modelos Universales.

*Todos los modelos estan mal pero algunos son útiles*. George Box (Box and Draper 1987)

Aunque probemos nuestro modelo y lo validemos con los métodos adecuados para obtener cierta capacidad de generalización en cierto dominio de aplicación, este se desempeñara pobre en otros dominios diferentes.



## Unidad 2 Interpretación Geométrica de la Clasificación

### Definición de clases

## Unidad 3 Clasificación No supervisada

### Definición y formulación matemática
Definición




## Bibliografía

1. https://www.aprendemachinelearning.com/7-pasos-machine-learning-construir-maquina/

https://www.um.es/geograf/sigmur/temariohtml/node74_mn.html

https://www.aprendemachinelearning.com/7-pasos-machine-learning-construir-maquina/

https://en.wikipedia.org/wiki/Cross-validation_(statistics)#/media/File:K-fold_cross_validation_EN.svg

https://quimicayalgomas.com/wp-content/uploads/2015/03/logaritmos-propiedades.png








