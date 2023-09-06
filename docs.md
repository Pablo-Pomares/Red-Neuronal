# Documentación de *network*

## Requisitos

Para el correcto funcionamiento de *network* se requiere de las librerías,

- *random*
- *numpy*
- *mnist_loader* ; que a su vez requiere de,
  - *pickle*
  - *gzip*

## *mnist_loader*

*mnist_loader* está compuesto de dos funciones principales: `load_data()` y `load_data_wrapper()`, aunque nosotros únicamente accederemos a la última.

`load_data()`: recupera del directorio *data* el archivo *mnist.pkl.gz*. Para adaptarlo a *Python* 3 se le tuvo que agregar que recuperara los datos en "latin1" en lugar de "ASCII" que está por default. Dichos datos son después divididos en "training_data", "validation_data" y "test_data".

`load_data_wrapper()`: reacomoda y divide los datos en inputs y outputs para su lectura en *network*. Para adaptarlo se tuvo que transformar en listas los zips.

Adicionalmente, hay una tercera función llamada `vectorized_result()` el cual sirve para ordenar los datos en `load_data_wrapper()`.

## *network*

Esta librería está compuesta únicamente de la clase `Network()`, la función sigmoide y su derivada.

Dentro de `Network()` se definen en `__init__()` las variables `weights` y `biases`, las cuales inicialmente están randomizadas. Después se definen las funciones:

### `SGD()`

Se toman los argumentos,

- `training_data`: datos de entrenamiento
- `epochs`: número de épocas
- `mini_batch_size`: tamaño de mini batch
- `eta`: tamaño del paso
- `test_data`: datos para la prueba

Lo que hace la función es tomar aleatoriamente los datos y crear los mini batches. Posteriormente, llama a la función `update_mini_batch()` e imprime el número de época junto con la certeza de la red.

### `update_mini_batch()`

Toma los argumentos `eta` y `mini_batch` y actualiza los argumentos `self.biases` y `self.weights`. Primero, llama a los argumentos `nabla_b` y `nabla_w` como los los sesgos y los pesos respectivamente. Después, define `delta_nabla_b` y `delta_nabla_w` a través de la función `backprop()`, que se encarga del algoritmo *backpropagation* para posteriormente actualizar `nabla_b` y `nabla_w` sumándoles sus respectivas deltas. Por último, actualiza los nuevos sesgos y pesos con el *gradient decent*,

$$
w_k \rightarrow w'_k = w_k - \frac{\eta}{n} \Delta w_k \\ \ \\
b_k \rightarrow b'_k = b_k - \frac{\eta}{n} \Delta b_k
$$

### `backprop()`

Como se mencionó anteriormente `backprop()` se encarga del algoritmo *backpropagation*. Empieza definiendo una delta para la última capa, a través de la función menor `cost_deriative()`, Y prosigue a generar la delta correspondiente a cada capa,
$$
\delta^l = (w^{l+1}\delta^{l+1})\cdot \sigma '(z^l)
$$

Finalmente, regresa las nuevas nablas con,

$$
\nabla b = \delta^l \\ \ \\
\nabla w = \delta^l (a^{l-1})^T
$$

### Funciones menores

Dentro de `Network()` se definen además estas funciones,

`cost_derivative()`: regresa la derivada de nuestra función de costo respecto a $a_i^L$, que es $\sigma (\Sigma_k w_{ik}^L a_k^{L-1} + b_i^L)$, en nuestro caso es $y_i(x) - a_i^L$.

`feedfoward()`: regresa,
$$
a \rightarrow a'=\sigma(w^T a + b)
$$

`evaluate()`: regresa el número de veces que al red estuvo correcto en los datos de prueba.

Fuera de `Network()` se definen en `sigmoid()` y `sigmoid_prime()` la función sigmoide y su derivada respectivamente.
