**Tarea 6:  Clasificación de pixeles de imágenes con regresión logística.** (Parte práctica del éxamen). 

Examen 60%, ((presentación = 60% ). **Propuesta de entrega: 23 de Abril.**

1. (10%) Definir un problema de clasificación binaria de pixeles en una imagen (¿Qué se quiere clasificar? árboles, carretera, techos de edificios....). 

Fernando Nateras: 100%

Antonio Chacon: 100%

C.A. 100%



1. (10%) Explorar visualmente las muestras con una gráfica de dispersión (*scatter plot*) para evaluar la separabilidad de las clases. (Scatter plot acompañada de una conclusión o comentario sobre los datos observados).

Fernando Nateras: 100%

Antonio Chacon: 100%

C.A 

1. (10%) Seleccionar muestras de las dos clases de una imagen donde aproximadamente cada una de ellas tengan las los mismos elementos o vectores de entrada para evitar clases desbalanceadas. (Indicar Explícitamente la parte del código que realiza esta tarea).

   1. Por ejemplo, si las dos clases contienen diferente número de elementos (e.g., $|C_2| > |C_1|$|) hacer :
      1. $C_1 \text{: }|--entrenamiento\text{ }50\%--|--prueba\text{ }50\%--|$
      
      2. $C_2\text{: }|--entrenamiento\text{ }50\%--|--prueba\text{ }50\%--|--descartar--|$
      
         
      
         Fernando Nateras: 100%
      
         Antonio Chacón: 100%
      
         C.A 100%

2. (10%) Construcción y validación visual de los conjuntos de entrenamiento y prueba.

   1. *Revolver* los vectores de cada clase $C_1$ y $C_2$ aleatoriamente ( con```numpy.random.shuffle```https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html) para construir conjuntos de entrenamiento y prueba representativo de cada clase. (mostrar una pequeña muestra de los primeros 5 valores del vector aleatorizado). 

      Fernando Nateras: 100%

      AC 100%

   2. Comparar la distribucion de los valores de los pixeles con  histogramas univariados (uno para cada banda de color de cada conjunto de cada clase (3x2x2=12 histogramas) ). Ejemplo: para los 2 conjuntos de C1 de entrenamiento y prueba,  comparar las distribuciones de cada color con un histograma   (Generar y reportar Total de 12 histogramas univariados: 3 vs 3 en para cada clase).

   Fernando Nateras;100%

   AC: 100%

   1. (5% extra, calcular media y covarianza multivariada para cada conjunto.). Tabla comparativa $\mu, \sum$
      1. https://numpy.org/doc/stable/reference/generated/numpy.cov.html

      2. https://www.geeksforgeeks.org/python-numpy-cov-function/,

      3. https://es.wikipedia.org/wiki/Covarianza

         FN(0%)

         AC 100%

3. (40%) Extender el modelo de regresión logística a tres dimensiones de la tarea 5 para clasificar las clases muestreadas de la imagen(FN: 100% AC 80%,). Hacer validación cruzadada de dos iteraciones (2 fold validation) y reportar cada iteración con una **matriz de confusión $^1$**, así como su precisión, exactitud, sensibilidad y especificidad. (Código y Reportar matriz de confusión). FN(0%)

    

    

    https://es.m.wikipedia.org/wiki/Archivo:Matriz_confusion.png. 

4. (20%) Generar una imagen binaria como resultado de clasificar el resto de los pixeles con el modelo entrenado. (Reportar Imagen). (FN: 0%)

Entregar el trabajo en Libreta de Jupyter. Describir detalladamente el procedimiento.

^1$**Matriz de confusión**

Muestra cuando una clase es confundida por otra por algún modelo de clasificación. Funciona cuando las clases están balanceadas.

Sirve para evaluar el desempeño de un modelo clasificador mas allá del cálculo de la precisión. (clasificaciones correctas / número total de elementos).



La matriz de confusión es una tabla de valores:

|                          |      | Estimación  (predicción) | Estimación  (predicción) |
| ------------------------ | ---- | ------------------------ | ------------------------ |
|                          |      | C1                       | C2                       |
| Clasificacion  verdadera | C1   | VP                       | FP                       |
| Clasificacion  verdadera | C2   | FN                       | VN                       |

Donde:

**VP**: Verdaderos positivo. Número de estimaciones acertadas pertenencientes a la clase positiva ($C_1$).

**VN**: Verdadero negativo. Número de estimaciones acertadas pertenecientes a la clase negativa ($C_2$).

**FP**: Falso positivo. Error tipo I. Desacierto estimando la clase. 

**FN**: Falso negativo. Error tipo II. Desacierto estimando la clase.

**Precisión**

$P=\frac{VP}{VP+FP}$

**Exactitud**  (*Accuracy*)

$AC = \frac{VP+VN}{VP+VN+FP+FN}$

**Sensibilidad**. (Tasa de verdaderos positivos) : 

$TVP = \frac{VP}{VP+FN}$

**Especificidad** (Tasa de Verdaderos Negativos): 

$TVN=\frac{VN}{VN+FP}$

Precisión y sensibilidad indican que tan relevante son los resultados.