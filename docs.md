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

Dentro de `Network()` se definen en `__init__` las variables `weights` y `biases`, las cuales inicialmente están randomizadas. Después se definen las funciones:

`SGD`: se toman los argumentos,

- `training_data`
- `epochs`
- `mini_batch_size`
- `eta`
- `test_data`
