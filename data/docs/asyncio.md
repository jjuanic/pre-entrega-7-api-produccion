---
title: asyncio — Programación asíncrona con corutinas
category: concurrencia
library: asyncio
tags: async, await, corutinas, event loop, taskgroup, concurrencia
---

# asyncio — Programación asíncrona con corutinas

`asyncio` es el módulo de la librería estándar para escribir código concurrente con la sintaxis `async`/`await`. Está pensado para cargas de trabajo **limitadas por entrada/salida** (I/O-bound): llamadas HTTP, consultas a bases de datos, lectura de sockets o llamadas a APIs de modelos de lenguaje. Mientras una operación espera la respuesta de la red, el event loop aprovecha ese tiempo para avanzar con otras tareas.

No es la herramienta adecuada para tareas que consumen CPU de forma intensiva (procesar imágenes, cálculos numéricos pesados): una corutina que calcula sin pausas bloquea el event loop completo. Para eso conviene `concurrent.futures.ProcessPoolExecutor`.

## Corutinas y el event loop

Una función definida con `async def` es una **función corutina**. Llamarla no ejecuta su cuerpo: devuelve un objeto corutina que hay que esperar con `await` o programar en el event loop.

```python
import asyncio

async def saludar(nombre: str) -> str:
    await asyncio.sleep(1)  # simula una espera de I/O
    return f"Hola, {nombre}"

async def main():
    mensaje = await saludar("Ada")
    print(mensaje)

asyncio.run(main())
```

`asyncio.run()` es el punto de entrada recomendado: crea un event loop nuevo, ejecuta la corutina principal hasta que termina, cancela las tareas pendientes y cierra el loop. Debe llamarse una sola vez por programa y **no puede invocarse desde un event loop que ya está corriendo** (por ejemplo, dentro de un notebook de Jupyter, donde directamente se puede usar `await main()` en la celda).

Un error frecuente es olvidar el `await`: Python emite la advertencia `RuntimeWarning: coroutine '...' was never awaited` y la corutina nunca se ejecuta.

## Tareas: ejecutar corutinas en simultáneo

`await` ejecuta las corutinas una detrás de otra. Para que avancen en paralelo hay que envolverlas en **tareas**:

```python
async def main():
    tarea_a = asyncio.create_task(saludar("Ada"))
    tarea_b = asyncio.create_task(saludar("Alan"))
    print(await tarea_a, await tarea_b)  # tarda ~1 segundo, no 2
```

Hay que guardar una referencia a la tarea creada con `create_task`: el event loop solo mantiene referencias débiles y una tarea sin referencias puede ser recolectada antes de terminar.

### asyncio.gather

`asyncio.gather(*aws)` ejecuta varias corutinas concurrentemente y devuelve sus resultados **en el mismo orden** en que se pasaron. Con `return_exceptions=True` las excepciones se devuelven como valores en la lista de resultados en lugar de propagarse, útil cuando se quiere procesar todos los resultados aunque algunos fallen.

### TaskGroup (Python 3.11+)

`asyncio.TaskGroup` es la alternativa moderna y más segura a `gather`. Implementa **concurrencia estructurada**: si una tarea del grupo falla, las demás se cancelan automáticamente y los errores se reportan juntos en un `ExceptionGroup`, que se puede manejar con `except*`.

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(saludar("Ada"))
        t2 = tg.create_task(saludar("Grace"))
    # al salir del bloque, todas las tareas terminaron
    print(t1.result(), t2.result())
```

## Timeouts y cancelación

Nunca conviene esperar indefinidamente una respuesta de red. `asyncio.timeout()` (3.11+) es un context manager que cancela el bloque si se excede el tiempo y lanza `TimeoutError`:

```python
async with asyncio.timeout(10):
    respuesta = await llamar_api()
```

En versiones anteriores se usa `asyncio.wait_for(corutina, timeout=10)`. Cuando una tarea se cancela con `tarea.cancel()`, dentro de la corutina se lanza `asyncio.CancelledError`. Se puede capturar para liberar recursos, pero debe volver a lanzarse: tragarse la cancelación rompe la semántica de `TaskGroup` y de los timeouts.

## Limitar la concurrencia con Semaphore

Lanzar mil peticiones simultáneas a una API suele terminar en errores 429 (rate limit). `asyncio.Semaphore(n)` permite que como máximo `n` corutinas entren a una sección al mismo tiempo; el resto espera su turno:

```python
semaforo = asyncio.Semaphore(5)

async def descargar(url):
    async with semaforo:  # como máximo 5 descargas en simultáneo
        return await cliente.get(url)
```

## Colas: productor-consumidor

`asyncio.Queue` coordina productores y consumidores sin locks explícitos. `await cola.put(item)` agrega un elemento (y espera si la cola tiene `maxsize` y está llena); `await cola.get()` espera hasta que haya un elemento. Combinando `cola.task_done()` con `await cola.join()` se puede esperar a que todos los elementos hayan sido procesados.

## Integrar código bloqueante

Si hay que llamar a una función síncrona y bloqueante (una librería sin soporte async, lectura de un archivo grande), no hay que llamarla directamente desde una corutina. `await asyncio.to_thread(funcion, *args)` la ejecuta en un hilo aparte y devuelve el resultado sin bloquear el event loop. Para mayor control existe `loop.run_in_executor(executor, funcion, *args)`.

## Modo debug

El modo debug de asyncio detecta corutinas nunca esperadas, callbacks lentos que bloquean el loop (más de 100 ms por defecto) y llamadas no thread-safe. Se activa con `asyncio.run(main(), debug=True)`, con la variable de entorno `PYTHONASYNCIODEBUG=1` o ejecutando Python con `-X dev`.
