---
title: logging — Registro de eventos de una aplicación
category: observabilidad
library: logging
tags: logs, handlers, niveles, rotacion, observabilidad
---

# logging — Registro de eventos de una aplicación

El módulo `logging` es el sistema estándar de Python para registrar lo que ocurre en una aplicación. A diferencia de `print`, permite clasificar los mensajes por **nivel de severidad**, enviarlos a distintos destinos (consola, archivos, servicios externos) con distintos formatos, y activarlos o silenciarlos sin tocar el código.

## Niveles de severidad

De menor a mayor: `DEBUG` (10), `INFO` (20), `WARNING` (30), `ERROR` (40) y `CRITICAL` (50). Un logger solo procesa los mensajes cuyo nivel es **igual o superior** al nivel configurado. El nivel por defecto del logger raíz es `WARNING`, por eso al empezar es común que los `logger.info(...)` "no aparezcan".

## Configuración rápida con basicConfig

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("Servidor iniciado en el puerto %d", 8000)
```

`basicConfig` configura el logger raíz y **solo tiene efecto la primera vez** que se llama (si ya hay handlers configurados, no hace nada, salvo que se pase `force=True`). Debe llamarse una única vez, en el punto de entrada de la aplicación, nunca dentro de una librería.

## Loggers jerárquicos: getLogger(__name__)

La convención es crear un logger por módulo con `logging.getLogger(__name__)`. Los nombres forman una jerarquía separada por puntos: el logger `app.db` es hijo de `app`. Los mensajes **se propagan** hacia los loggers padres hasta el raíz, así que alcanza con configurar handlers en la raíz o en `app`. Esto permite, por ejemplo, subir a `DEBUG` solo el logger `app.db` mientras el resto queda en `INFO`. Para cortar la propagación de un logger se usa `logger.propagate = False`.

## Arquitectura: loggers, handlers, formatters y filters

- **Logger**: el objeto con el que el código emite mensajes (`logger.warning(...)`).
- **Handler**: decide **a dónde** va cada mensaje. Un logger puede tener varios, cada uno con su propio nivel: por ejemplo, `INFO` a consola y `ERROR` a un archivo.
- **Formatter**: decide **cómo** se ve el mensaje. Atributos útiles: `%(asctime)s`, `%(levelname)s`, `%(name)s`, `%(funcName)s`, `%(lineno)d`, `%(process)d`.
- **Filter**: permite descartar o enriquecer registros con lógica propia (por ejemplo, agregar un `request_id`).

Handlers más usados: `StreamHandler` (consola), `FileHandler` (archivo), `RotatingFileHandler`, `TimedRotatingFileHandler`, `SMTPHandler` (mail) y `QueueHandler` (para no bloquear el hilo principal mientras se escribe el log).

## Rotación de archivos de log

Un archivo de log que crece sin límite termina llenando el disco. El módulo `logging.handlers` ofrece dos estrategias de rotación:

```python
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Rota cuando el archivo supera 5 MB y conserva 3 respaldos: app.log.1, app.log.2, app.log.3
por_tamanio = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")

# Rota todos los días a medianoche y conserva una semana de historial
por_tiempo = TimedRotatingFileHandler("app.log", when="midnight", backupCount=7, encoding="utf-8")
```

## Registrar excepciones

Dentro de un bloque `except`, `logger.exception("mensaje")` registra el mensaje con nivel `ERROR` **e incluye el traceback completo** de la excepción. Es equivalente a `logger.error("mensaje", exc_info=True)`. Registrar solo `str(e)` pierde la información de en qué línea ocurrió el error.

```python
try:
    procesar_pedido(pedido)
except Exception:
    logger.exception("Falló el procesamiento del pedido %s", pedido.id)
    raise
```

## Formato diferido de mensajes

Conviene pasar los valores como argumentos (`logger.debug("Usuario %s", usuario)`) en lugar de usar f-strings (`logger.debug(f"Usuario {usuario}")`). Con argumentos, el string solo se formatea si el mensaje realmente se va a emitir: si el nivel `DEBUG` está desactivado, se ahorra el costo de formatear. Además, herramientas como Sentry agrupan mejor los mensajes que comparten la misma plantilla.

## Configuración declarativa con dictConfig

En aplicaciones medianas o grandes la configuración se define como un diccionario (que puede venir de un archivo YAML o JSON) y se aplica con `logging.config.dictConfig(config)`. Así se separa la configuración del código y se pueden cambiar niveles y destinos por entorno (desarrollo, staging, producción). Un detalle importante: por defecto `dictConfig` deshabilita los loggers ya existentes; para evitarlo se incluye `"disable_existing_loggers": False`.

## Buenas prácticas en librerías

Una librería nunca debe llamar a `basicConfig` ni agregar handlers propios a la raíz: solo crea sus loggers con `getLogger(__name__)` y, como mucho, agrega un `logging.NullHandler()` a su logger principal. La decisión de qué mostrar y dónde es de la aplicación que usa la librería.
