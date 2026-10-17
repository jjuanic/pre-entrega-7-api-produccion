---
title: dataclasses — Clases de datos sin código repetitivo
category: modelado-de-datos
library: dataclasses
tags: dataclass, clases, inmutabilidad, default_factory, slots
---

# dataclasses — Clases de datos sin código repetitivo

El decorador `@dataclass` genera automáticamente los métodos especiales de una clase cuyo propósito principal es **almacenar datos**: `__init__`, `__repr__` y `__eq__`, a partir de los atributos declarados con anotaciones de tipo.

```python
from dataclasses import dataclass

@dataclass
class Producto:
    nombre: str
    precio: float
    stock: int = 0

p = Producto("Teclado", 25_000.0)
print(p)  # Producto(nombre='Teclado', precio=25000.0, stock=0)
print(p == Producto("Teclado", 25_000.0))  # True
```

Las anotaciones de tipo son obligatorias para que un atributo se considere campo, pero **no se validan en tiempo de ejecución**: `Producto("Mouse", "caro")` no lanza error. Si se necesita validación, conviene usar Pydantic o validar en `__post_init__`.

## Valores por defecto mutables: default_factory

Los campos con valor por defecto deben ir después de los campos sin valor por defecto. Además, `dataclasses` **prohíbe usar listas, diccionarios o sets como valor por defecto** directo, porque esa misma instancia sería compartida por todos los objetos. Escribir `etiquetas: list = []` lanza `ValueError: mutable default <class 'list'> for field etiquetas is not allowed: use default_factory`.

La solución es `field(default_factory=...)`, que llama a la función indicada para crear un valor nuevo en cada instancia:

```python
from dataclasses import dataclass, field

@dataclass
class Pedido:
    cliente: str
    items: list[str] = field(default_factory=list)
    metadatos: dict[str, str] = field(default_factory=dict)
```

`field()` acepta otras opciones útiles: `repr=False` oculta el campo en el `__repr__` (por ejemplo, para contraseñas o tokens), `compare=False` lo excluye de las comparaciones e `init=False` lo excluye del constructor.

## Validación y campos derivados: __post_init__

El `__init__` generado llama a `__post_init__` (si existe) al final. Es el lugar para validar datos o calcular campos derivados:

```python
@dataclass
class Rango:
    minimo: int
    maximo: int
    amplitud: int = field(init=False)

    def __post_init__(self):
        if self.minimo > self.maximo:
            raise ValueError("minimo no puede ser mayor que maximo")
        self.amplitud = self.maximo - self.minimo
```

## Instancias inmutables: frozen=True

Con `@dataclass(frozen=True)` cualquier intento de asignar un atributo después de crear el objeto lanza `dataclasses.FrozenInstanceError`. Las instancias congeladas (si sus campos también son hashables) generan `__hash__` y pueden usarse como claves de diccionario o elementos de un set. Para "modificar" un objeto inmutable se crea una copia con cambios usando `dataclasses.replace(obj, campo=nuevo_valor)`.

## Otras opciones del decorador

- `order=True`: genera `__lt__`, `__le__`, `__gt__` y `__ge__`, comparando los campos como si fueran una tupla en el orden de declaración.
- `slots=True` (3.10+): genera `__slots__`, lo que reduce el consumo de memoria por instancia y hace más rápido el acceso a atributos; a cambio, no se pueden agregar atributos dinámicamente.
- `kw_only=True` (3.10+): obliga a pasar todos los campos por nombre. También se puede usar el marcador `KW_ONLY` para que solo los campos posteriores sean keyword-only, lo que resuelve el problema de mezclar campos con y sin valor por defecto en herencia.

## Utilidades: asdict, astuple, fields

`dataclasses.asdict(obj)` convierte la instancia en un diccionario de forma **recursiva** (incluye dataclasses anidadas, listas y diccionarios), lo que facilita serializarla a JSON con `json.dumps(asdict(obj))`. `astuple(obj)` hace lo mismo pero devuelve una tupla. `fields(obj)` devuelve la descripción de cada campo (nombre, tipo, valor por defecto), útil para introspección.

## dataclass, NamedTuple, TypedDict o Pydantic

- **NamedTuple**: inmutable y se comporta como tupla (se puede desempaquetar e indexar). Útil para valores de retorno livianos.
- **TypedDict**: solo describe la forma de un diccionario para el chequeo estático de tipos; en tiempo de ejecución sigue siendo un `dict` común.
- **dataclass**: la opción estándar para objetos de dominio mutables o inmutables, sin dependencias externas.
- **Pydantic**: cuando los datos vienen de afuera (JSON de una API, respuesta de un LLM) y hay que validarlos y convertir tipos en tiempo de ejecución.
