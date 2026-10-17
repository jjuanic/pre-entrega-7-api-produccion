---
title: concurrent.futures — Hilos y procesos con una interfaz común
category: concurrencia
library: concurrent.futures
tags: threads, procesos, gil, executor, paralelismo
---

# concurrent.futures — Hilos y procesos con una interfaz común

El módulo `concurrent.futures` ofrece una interfaz de alto nivel para ejecutar funciones de forma asíncrona usando **pools de hilos** o **pools de procesos**. Las dos implementaciones, `ThreadPoolExecutor` y `ProcessPoolExecutor`, comparten la misma API (la clase base `Executor`), así que pasar de una a otra suele requerir cambiar una sola línea.

## El GIL y la elección entre hilos y procesos

En CPython, el **Global Interpreter Lock (GIL)** permite que un solo hilo ejecute bytecode de Python a la vez dentro de un proceso. Esto define cuándo conviene cada executor:

- **ThreadPoolExecutor**: ideal para tareas **I/O-bound** (descargas, consultas a APIs, lectura de archivos). Mientras un hilo espera la red, libera el GIL y otro hilo puede avanzar. Los hilos son livianos y comparten memoria.
- **ProcessPoolExecutor**: ideal para tareas **CPU-bound** (cálculos, compresión, parseo intensivo). Cada proceso tiene su propio intérprete y su propio GIL, por lo que se logra paralelismo real en varios núcleos. A cambio, crear procesos es más costoso y los argumentos y resultados deben poder serializarse con `pickle`.

Desde Python 3.13 existe una compilación experimental "free-threaded" sin GIL, pero la versión estándar del intérprete sigue teniéndolo.

## Uso básico

```python
from concurrent.futures import ThreadPoolExecutor
import urllib.request

def descargar(url: str) -> int:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return len(resp.read())

urls = ["https://example.com", "https://python.org"]

with ThreadPoolExecutor(max_workers=8) as executor:
    tamanios = list(executor.map(descargar, urls))
```

Usar el executor como context manager (`with`) garantiza que al salir del bloque se espere a que terminen todas las tareas y se liberen los workers (equivale a llamar a `executor.shutdown(wait=True)`).

## submit, Future y as_completed

`executor.submit(funcion, *args)` programa una llamada y devuelve inmediatamente un objeto **Future**, que representa un resultado que todavía no existe:

- `future.result(timeout=None)`: bloquea hasta obtener el valor; si la función lanzó una excepción, la vuelve a lanzar acá.
- `future.done()`: indica si ya terminó.
- `future.cancel()`: intenta cancelarla (solo funciona si todavía no empezó).
- `future.add_done_callback(fn)`: registra una función que se llama al terminar.

Para procesar resultados **a medida que van llegando**, en lugar de en el orden de envío, se usa `as_completed`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor() as executor:
    futuros = {executor.submit(descargar, u): u for u in urls}
    for futuro in as_completed(futuros):
        url = futuros[futuro]
        try:
            print(url, futuro.result())
        except Exception as exc:
            print(f"{url} falló: {exc}")
```

`executor.map` es más simple pero devuelve los resultados en el orden original y, si una llamada falla, la excepción aparece recién al iterar ese resultado. `wait(futuros, return_when=FIRST_COMPLETED)` permite esperar con más control.

## Cantidad de workers

Si no se indica `max_workers`, `ThreadPoolExecutor` usa `min(32, os.cpu_count() + 4)` hilos y `ProcessPoolExecutor` usa la cantidad de CPUs disponibles. Para I/O-bound se puede subir bastante el número de hilos; para CPU-bound, usar más procesos que núcleos no aporta y suma overhead.

## ProcessPoolExecutor en Windows y macOS

En Windows y macOS los procesos hijos se crean con el método `spawn`: cada uno vuelve a importar el módulo principal. Por eso, el código que crea el pool **debe estar protegido** con `if __name__ == "__main__":`, o el programa intentará crear procesos de forma recursiva y fallará. Además, la función que se envía debe estar definida a nivel de módulo (las lambdas y funciones anidadas no se pueden serializar con pickle).

```python
from concurrent.futures import ProcessPoolExecutor

def es_primo(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        resultados = list(executor.map(es_primo, range(10_000), chunksize=500))
```

El parámetro `chunksize` de `map` agrupa elementos en lotes para reducir el costo de comunicación entre procesos; con iterables grandes mejora mucho el rendimiento.

## Sincronización con hilos

Aunque el GIL existe, las operaciones compuestas como `contador += 1` **no son atómicas**: dos hilos pueden leer el mismo valor y perder una actualización. Para proteger estado compartido se usa `threading.Lock`:

```python
import threading

lock = threading.Lock()
with lock:
    contador += 1
```

En muchos casos es mejor evitar el estado compartido y que cada tarea devuelva su resultado a través del Future.

## Relación con asyncio

Desde una corutina se puede delegar trabajo bloqueante a un executor con `loop.run_in_executor(executor, funcion, *args)`, o con el atajo `asyncio.to_thread`. Así se combinan ambos modelos: asyncio para coordinar I/O y un pool de procesos para el cálculo pesado.
