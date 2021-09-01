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









## 8.3 Herramientas OBIA

