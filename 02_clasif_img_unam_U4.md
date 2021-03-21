## Unidad 4 Clasificación Supervisada

Orden de los temas

1. Cálculo de parámetros estadísticos

2. Clasificación

3. Análisis de las muestras

4. Selección de *áreas* de entrenamiento

   

Hasta el momento hemos visto: 

La interpretación geométrica de un clasificador lineal $\mathbf{wx}+w_0 = 0$ con valores dados a los parámetros $\theta = (\mathbf{w},w_0)$.

Diferentes formulaciones de la función objetivo de la regresión logística comentando sus ventajas e inconvenientes. (Linea, sigmoide y log-sigmoide)

¿Qué falta ahora?

Dado un conjunto de datos previamente clasificados encontrar el valor de los parámetros $\theta$ que optimice la función objetivo. 

Para esto tomamos su interpretación estadística.

 La regresión logística supone que para un dato $(\mathbf{x}_i,y_i)$, ($\mathbf{x}_i \in \mathbf{R}^{Dim}, y_i \in \{0,1\}$)tomado de un conjunto de datos independientemente distribuidos de una misma distribución de probabilidad, y linealmente separables, existe una probabilidad de que  $\mathbf{x}_i$ pertenezca a una clase $y_i=1$ dada por una función sigmoidal. 





$p(y_i=1|\mathbf{x}_i)=sigm(z)$

donde $z = \sum_{i=1}^m w_ix_i +w_0$

y por la ley de la probabilidad $p(y_i=1|\mathbf{X=x}_i) + p(y_i=0|\mathbf{X=x}_1) = 1$

$p(y_i=0|\mathbf{X=x}_i)=1 - sigm(z)$



Utilizando esta suposición es posible utilizar o crear un algoritmo que a partir de un conjunto de datos clasificados de manera binaria   $$D = \{(\mathbf{x}_i,y_i)\}^N_{i=1}$$,  obtener los mejores valores de los parámetros para maximizar la precisión de nuestro clasificador. 



Una manera es formulando una función de costo $C(\theta)$ a minimizar (e.g, minimizar el error de clasificación ), o en el caso estadístico, maximizando la máxima verosimilitud. Esto se hace en dos pasos generales.

1. Formulando la función de verosimilitud
2. Encontrar los valores $\mathbf{w}$ que maximizan la función de verosimilitud.

Equivalentemente en optimización 

1. Formulando una función de costo
2. Encontrar los parámetros $\mathbf{w} $ que minimicen la función de costo.



Como las variables a clasificar son binarias, la salida de la función logística se interpreta como probabilidad de pertenencia a una clase. Por lo tanto podemos interpretar cada etiqueta como una variable aleatoria de Bernoulli $Y \sim Ber(p)$ donde $ p=sigm(\mathbf{w}^T\mathbf{x})$  (lo siento si veremos algo de Bernoulli :O!! ).



Entonces utilizando la función de probabilidad de Bernoulli evaluamos en una sola expresión la probabilidad de que un dato pertenezca a una clase o a otra indistintamente de su etiqueta. 

$P(Y=y_i|\mathbf{X}=\mathbf{x}_i) = sigm(\mathbf{w}^T\mathbf{x}_i +w_0)^y \cdot [1-sigm(\mathbf{w}^T\mathbf{x}_i)]^{(1-y)}$

Analicemos con detenimiento los exponentes de esta función. 

Si $y=0$ entonces la primera parte de la ecuación sería 1 multiplicado por la probabilidad de que $y=0$.

Si $y=1$ entonces la primera parte de la ecuación sería _____

Que multiplica a ______



La verosimilitud es una función que depende de los parámetros de un modelo estadístico para realizar buenas inferencias acerca de su valor a partir de un conjunto de observaciones (Wikipedia). En términos prácticos, sirve también como una medida de desempeño de un conjunto de parámetros $\theta$ considerando el conjunto de  datos $D$

La verosimilitud se puede expresar de manera genera como:

 $L(\theta)  =  \prod_{i=1}^m p(Y=y_i | X= \mathbf{x}_i,\theta) $ 

y con una simple sustitución por la función de probabilidad de Bernoulli expresión tenemos:

 $L(\theta)  = \prod_{i=1}^m sigm(\mathbf{w}^T\mathbf{x}_i +w_0)^{y_i} \cdot [1-sigm(\mathbf{w}^T\mathbf{x}_i)]^{(1-{y_i})} $ 



Esta función ya nos sirve para utilizarla en nuestro problema de optimización. Ya nos arroja información útil. Pero podemos simplificarla por razones de estabilidad numérica y remover componentes exponenciales que no contribuyen, por lo tanto formulamos la log-verosimilitud.

 $log (L(\theta)) = LL(\theta)$ 



Esta expresión se escribe:

$LL(\theta) = log (\prod_{i=1}^m  sigm(\mathbf{w}^T\mathbf{x}_i +w_0)^{y_i} \cdot [1-sigm(\mathbf{w}^T\mathbf{x}_i)]^{(1-{y_i})}) $ 



Qué paso sigue? **Acordarse de las propiedades de los Logaritmos!!!.**

$LL(\theta) = \sum_{i=1}^m (y_i \text{ log } sigm(\mathbf{w}^T \mathbf{x}_i) + (1-y_i) \text{ log } [1-sigm(\mathbf{w}^T \mathbf{x}_i)])$

Tarea! Aplicar el logaritmo a lo que falta.



Esta expresión la usaremos para construir nuestra función objetivo a maximizar.

$\mathbf{w},w_0 = \text{arg max }_{\theta = (\mathbf{w},w_0)} LL(\theta)$

Cómo podemos maximizar?

1. Probando por fuerza bruta diferentes parámetros de $\theta$ y quedarnos con aquellos que nos den el máximo $LL(\theta)$. Muy costoso computacionalmente hablando.
2. Asumir que $LL(\theta)$ tiene una estructura numérica que podemos utilizar a nuestro favor:  es monótona creciente y tiene al menos un óptimo local ( un punto donde los parámetros $\theta$ ya no mejoran, o sea que la pendiente en la función evaluada en $\hat{\theta}$  = 0). De manera mas elegante es donde su derivada o gradiente $  \nabla  LL(\theta)  = 0$

$\nabla f=\left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots \frac{\partial f}{\partial x_n} \right]$



Desafortunadamente no existe una expresión exacta que resuelva 

$ \nabla  LL(\theta)  = 0$

Por lo tanto procedemos a utilizar optimización numérica para aproximarnos a la solución.



Seleccionamos el algoritmo de gradiente ascendente para maximizar, (descendente para minimizar $-LL(\theta)$), la función $LL(\theta)$. 

Para esto requerimos la derivada parcial $\frac{\partial LL(\theta)}{\partial \theta_j} $ para cada parámetro que es la que  dirigirá los parámetros hacia el óptimo.

$\frac{\partial LL(\theta)}{\partial \theta_j} = \sum_{i=1}^n [y_i-sigm(\theta^T\mathbf{x}_i)]x_j$





La estructura básica de proceso de optimización es:

$\theta_{t+1} = \theta_{t} + \text{(tamaño de paso) }* \text{dirección}$



Los parámetros $\theta$ se inicializan con algún valor dentro del rango posible, de preferencia cerca de la solución si se conoce.

El tamaño de paso acelera/desacelera el acercamiento al óptimo, pero puede inducir a la no convergencia.

La dirección la proporciona ....??



$\theta_{j,t+1} = \theta_{j,t}+n \cdot \frac{\partial LL (\theta_t)}{\partial \theta_{j,t}}$





Algoritmo en pseudo código



* Inicializar parámetros $\{\theta_j=0\}_{j=1}^m $

* Repetir varias veces hasta cumplir un criterio de parada

  * Inicializar gradiente $\{grad_j=0\}_{j=1}^m $

  * Para cada tupla $(x_i,y_i)$:

    * para cada parámetro $j \in \{1,\dots,m\}$
      * $grad_{j} =  [y_i-sigm(\theta^T\mathbf{x}_i)]x_j$

  * $\theta_j = \theta + n * grad_j$ para $j \in {1,\dots,m}$

    

en el caso de considerar $sigm(\theta^T\mathbf{x}_i + \theta_0)$, en vez de utilizar la derivada explicita se define una característica especial $x_0=1$ para obtener $sigm(\theta^T\mathbf{x}_i)= \frac{1}{1+exp{-(\theta^T\mathbf{x}_i+x_0\theta_0)} }$







## Bibliografía





https://www.um.es/geograf/sigmur/temariohtml/node74_mn.html

https://turing.iimas.unam.mx/~ivanvladimir/posts/gradient_descent/

https://turing.iimas.unam.mx/~ivanvladimir/posts/gradient_descent/