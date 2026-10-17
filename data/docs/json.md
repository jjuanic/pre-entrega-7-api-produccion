---
title: json — Serialización y parseo de JSON
category: serializacion
library: json
tags: json, serializacion, parseo, encoding, apis
---

# json — Serialización y parseo de JSON

El módulo `json` convierte objetos de Python a texto JSON (**serializar**) y texto JSON a objetos de Python (**parsear** o deserializar). Es la forma estándar de intercambiar datos con APIs web, archivos de configuración y respuestas estructuradas de modelos de lenguaje.

## Las cuatro funciones principales

- `json.dumps(obj)`: objeto de Python → string JSON.
- `json.loads(texto)`: string JSON → objeto de Python.
- `json.dump(obj, archivo)`: escribe el JSON directamente en un archivo abierto.
- `json.load(archivo)`: lee y parsea el JSON de un archivo abierto.

Una regla mnemotécnica: las versiones con **s** final trabajan con **s**trings.

```python
import json

pedido = {"id": 42, "cliente": "Ana", "items": ["teclado", "mouse"], "pagado": True}
texto = json.dumps(pedido)
datos = json.loads(texto)

with open("pedido.json", "w", encoding="utf-8") as f:
    json.dump(pedido, f, ensure_ascii=False, indent=2)
```

## Tabla de conversión de tipos

| Python | JSON |
|---|---|
| `dict` | objeto |
| `list`, `tuple` | array |
| `str` | string |
| `int`, `float` | number |
| `True` / `False` | `true` / `false` |
| `None` | `null` |

La conversión no es simétrica: una tupla se serializa como array y al volver se obtiene una **lista**. Las claves de los diccionarios en JSON siempre son strings, así que `{1: "a"}` vuelve como `{"1": "a"}`.

## Parámetros de formato

- `indent=2`: genera JSON legible, con saltos de línea e indentación. Sin `indent` se obtiene una sola línea compacta.
- `ensure_ascii=False`: por defecto (`ensure_ascii=True`) todos los caracteres no ASCII se escapan, de modo que `"canción"` se escribe como `"canción"`. Con `ensure_ascii=False` los acentos, eñes y emojis se escriben tal cual; en ese caso hay que abrir el archivo con `encoding="utf-8"`.
- `sort_keys=True`: ordena las claves alfabéticamente, útil para comparar archivos o generar hashes estables del contenido.
- `separators=(",", ":")`: elimina los espacios después de comas y dos puntos para obtener la representación más compacta posible.

## Serializar tipos que JSON no soporta

Intentar serializar un `datetime`, un `Decimal`, un `set` o un objeto propio lanza `TypeError: Object of type datetime is not JSON serializable`. Hay dos formas de resolverlo.

Con el parámetro `default`, una función que se llama para cada objeto que el encoder no sabe convertir:

```python
from datetime import datetime
from decimal import Decimal

def convertir(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (Decimal, set)):
        return float(obj) if isinstance(obj, Decimal) else sorted(obj)
    raise TypeError(f"No serializable: {type(obj).__name__}")

json.dumps({"fecha": datetime.now(), "total": Decimal("10.50")}, default=convertir)
```

O creando una subclase de `json.JSONEncoder` que sobrescriba el método `default` y pasándola con `cls=MiEncoder`. Para dataclasses, lo más simple es `json.dumps(dataclasses.asdict(obj))`.

## Personalizar el parseo

- `object_hook`: función que recibe cada diccionario parseado y puede devolver otro objeto (por ejemplo, convertir dicts en instancias de una clase).
- `parse_float=Decimal`: parsea los números con decimales como `Decimal` en lugar de `float`, evitando errores de redondeo en montos de dinero.
- `object_pairs_hook`: recibe la lista de pares clave-valor; útil para detectar claves duplicadas, que por defecto se resuelven en silencio quedándose con la última.

## Manejo de errores: JSONDecodeError

Si el texto no es JSON válido, `loads` lanza `json.JSONDecodeError` (subclase de `ValueError`), que informa el mensaje, la línea (`lineno`), la columna (`colno`) y la posición del error. Es el error típico cuando un LLM devuelve el JSON envuelto en un bloque de markdown o con texto extra antes y después:

```python
try:
    datos = json.loads(respuesta_del_modelo)
except json.JSONDecodeError as e:
    print(f"JSON inválido en línea {e.lineno}, columna {e.colno}: {e.msg}")
```

Errores frecuentes que JSON no admite: comillas simples, comas finales después del último elemento, comentarios, y los valores `NaN` o `Infinity` (Python los acepta y los genera por defecto; con `allow_nan=False` se rechazan para producir JSON estrictamente válido).

## Herramienta de línea de comandos

El módulo incluye un validador y formateador: `python -m json.tool archivo.json` muestra el JSON indentado o informa el error de sintaxis. Acepta opciones como `--sort-keys`, `--indent` y `--compact`, y puede leer desde la entrada estándar: `curl -s https://api.ejemplo.com/datos | python -m json.tool`.
