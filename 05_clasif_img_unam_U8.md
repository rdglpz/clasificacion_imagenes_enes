# Unidad 8 . Clasificación Orientada a Objetos

## 

## 8.1 Introducción al análisis de Imágenes Orientado a Objetos

## Clasificación de Contexto



* En la vida diaria utilizamos el contexto para reconocer patrones.
* Sin el contexto sería complicado dar una interpretación correcta a un problema o situación.
* Clasificación de imágenes considerando su contexto, al contrario de la clasificación por pixel, la idea es considerar un grupo de pixeles que formen parte de una "entidad", donde esta entidad por ejemplo pueden ser caracteres, imágenes completas, etc. La entidad puede tener diferentes propiedades dependiendo del contexto. Al considerar el contexto, se introduce el efecto subjetivo de la percepción.  



![Different lines in different contexts](http://www.bic.mni.mcgill.ca/~mallar/CS-644B/lines.gif)

Las lineas que vemos arriba, todas tienen la misma longitud. Sin embargo, segun el contexto (las flechas añadidas), estas pueden pareccer mas largas o cortas.

Se le llama ilusión de Muller Lyer .[Toussaint](http://www.bic.mni.mcgill.ca/~mallar/CS-644B/ref.html)

### Contexto en clasificación de imagenes.

Observa la animación. 



![img](http://www.bic.mni.mcgill.ca/~mallar/CS-644B/happyfaceanim.gif)

Reconocer esta imagen es complicado si no conocemos el contexto. 

Si conocemos el contexto es muy sencillo identificar de lo que se trata y clasificar la imagen.

Las herramientas que hemos utilizado junto con el enfoque de clasificación por pixeles, el contexto no sirve de mucho porque aunque conozcamos el contexto  los modelos vistos hasta ahora (con excepción de redes neuronales) son incapaces de considerar esta información en el aprendizaje.



Una manera de clasificar la imagen de acuerdo a su contexto es utilizar un enfoque de emparejamiento de la imagen con un "prototipo" (template ).

Comparar esta imgen con todos los prototipos disponibles, donde cada uno tiene una etiqueta asignada. El prototipo mas parecido con la imagen será la etiqueta asignada a la imagen.



Este método se llama búsqueda en la tabla,  ( *table look-up* o *dictionary look-up*).

Este método es muy ineficiente por la cantidad de operaciones que se incrementan con el número de combinaciones posibles. 



Soluciones.

* Usando una versión bidimensional de cadenas de Markov llamado Mallas de Markov

* ![Image87](http://www.bic.mni.mcgill.ca/~mallar/CS-644B/Image87.gif)

* Probabilidades marginales.

* **REDES NEURONALES**

## 8.2 Fundamentos y aplicaciones. Metodología para la extracción de información

Antecedentes:

* Continuo incremento de la resolución de las imágenes satelitales.
* Demanda por automatizar su análisis.
* El reto es producir metodologías computacionales que ayuden a mejorar la interpretación de  un experto.
* Los metodos basados en objetos mejoran la clasificación basada en pixeles.
* Los objetos de imágenes pueden contener mas atributos que solo la información espectral.
* El procesamiento basado en objetos no solo considera la información contextual, si no la forma y las relaciones espaciales entre regiones en la imagen.



Un nuevo requirimiento en muchas aplicaciones de imágenes es la necesidad de crear mosaicos que representen una región definida por características similares. Estos pueden ser por ejemplo, características ecológicas, sociales, fisiográficas. 

La integración de datos adicionales se ha estudiado como una orientación de fusión de datos de multiples recursos y modelos de clasificación contextual para obtener una mejor precisión comparando con los enfoques basados en pixeles.

* human photo-interpreters also implicitly use structural knowledge in the manual classification process. This type of process has hardly been utilized in previous research.

* Diferentes enfoques en clasificacion de contexto han sido propuestos.



Metodologías existentes 

* Clasificación por pixel (*tradicional*).

  * Desventajas. No utiliza información espacial o contextual.

* Segmentación de imágenes.

  * Necesita información sobre la escala.
  * Ejemplo de esto son algoritmos de crecimiento de regiones.
  * Hay dos maneras de hacer segmentación 1) de manera manual por un experto 2) segmentación automática.

* Información de contexto y textura.
* * Descomponer una imagen y analizar la relación de sus elementos mas simples es una manera de incorporar información de contexto.
* Una de las primeras utilidades de utilizar el contexto, es para usar la información de las texturas en la clasificación.
*  



* Object-based image analysis compared with concept with traditional pixel-based image analysis.

PEr pixel methods:











## 8.3 Herramientas OBIA

