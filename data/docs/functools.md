---
title: functools — Funciones de orden superior y caché
category: programacion-funcional
library: functools
tags: decoradores, cache, memoizacion, lru_cache, partial
---

# functools — Funciones de orden superior y caché

El módulo `functools` reúne herramientas para trabajar con funciones como objetos: decoradores de caché, aplicación parcial de argumentos, despacho por tipo y utilidades para escribir decoradores correctamente.

## Memoización con lru_cache y cache

`@functools.lru_cache(maxsize=128)` guarda los resultados de una función según sus argumentos. Si se la llama de nuevo con los mismos argumentos, devuelve el resultado guardado sin volver a calcularlo. Cuando la caché se llena, descarta la entrada usada hace más tiempo (política *Least Recently Used*).

```python
from functools import lru_cache

@lru_cache(maxsize=256)
def fibonacci(n: int) -> int:
    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)

fibonacci(80)               # instantáneo gracias a la caché
print(fibonacci.cache_info())  # CacheInfo(hits=78, misses=81, maxsize=256, currsize=81)
fibonacci.cache_clear()     # vacía la caché
```

`@functools.cache` (Python 3.9+) es equivalente a `lru_cache(maxsize=None)`: una caché sin límite, más simple y un poco más rápida, pero que puede crecer indefinidamente.

Puntos importantes:

- Los argumentos deben ser **hashables**: no se puede cachear una función que recibe listas o diccionarios (se obtiene `TypeError: unhashable type`). Una solución es convertirlos a tuplas o `frozenset`.
- Solo tiene sentido en **funciones puras**: si el resultado depende de algo externo (la hora, una base de datos), la caché devolverá datos viejos.
- Con `typed=True`, `f(3)` y `f(3.0)` se cachean por separado.
- En métodos de instancia, la caché mantiene una referencia a `self` y puede impedir que el objeto sea liberado de memoria.

## cached_property

`@functools.cached_property` convierte un método en un atributo que se calcula **una sola vez por instancia** y luego queda guardado en el `__dict__` del objeto. Es útil para atributos costosos de calcular que no cambian:

```python
from functools import cached_property

class Dataset:
    def __init__(self, ruta):
        self.ruta = ruta

    @cached_property
    def filas(self):
        print("leyendo archivo...")
        return open(self.ruta).read().splitlines()
```

Para invalidar el valor se borra el atributo: `del dataset.filas`.

## partial: fijar argumentos

`functools.partial(funcion, *args, **kwargs)` crea una nueva función con algunos argumentos ya fijados. Es muy útil para adaptar funciones a interfaces que esperan menos parámetros, como callbacks o `map`:

```python
from functools import partial

def potencia(base, exponente):
    return base ** exponente

cuadrado = partial(potencia, exponente=2)
print(cuadrado(7))  # 49

convertir_binario = partial(int, base=2)
print(convertir_binario("1011"))  # 11
```

## wraps: escribir decoradores correctamente

Un decorador reemplaza la función original por una función envoltorio. Sin precauciones, la función decorada pierde su nombre, su docstring y sus anotaciones (`__name__` pasa a ser `"wrapper"`). `@functools.wraps(func)` copia esos metadatos al envoltorio:

```python
import time
from functools import wraps

def medir_tiempo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"{func.__name__} tardó {time.perf_counter() - inicio:.3f}s")
    return wrapper
```

Además, `wraps` agrega el atributo `__wrapped__`, que permite acceder a la función original (por ejemplo, para testearla sin el decorador).

## singledispatch: sobrecarga por tipo

`@functools.singledispatch` transforma una función en una **función genérica** que elige la implementación según el tipo del primer argumento:

```python
from functools import singledispatch

@singledispatch
def serializar(valor):
    raise TypeError(f"Tipo no soportado: {type(valor)}")

@serializar.register
def _(valor: int):
    return str(valor)

@serializar.register
def _(valor: list):
    return "[" + ", ".join(serializar(v) for v in valor) + "]"
```

Para métodos de clase existe `functools.singledispatchmethod`, que despacha según el tipo del primer argumento después de `self`.

## reduce y total_ordering

`functools.reduce(funcion, iterable, inicial)` aplica una función acumulativamente: `reduce(lambda a, b: a * b, [1, 2, 3, 4])` devuelve 24. Para sumas o máximos simples conviene usar `sum()` o `max()`, que son más legibles.

`@functools.total_ordering` completa los métodos de comparación de una clase: si se define `__eq__` y uno de `__lt__`, `__le__`, `__gt__` o `__ge__`, el decorador genera los demás automáticamente.
