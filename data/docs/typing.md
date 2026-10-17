---
title: typing — Anotaciones de tipos y chequeo estático
category: tipado
library: typing
tags: type hints, mypy, protocol, generics, typeddict
---

# typing — Anotaciones de tipos y chequeo estático

Las anotaciones de tipos (*type hints*) documentan qué tipos espera y devuelve cada función. Python **no las verifica en tiempo de ejecución**: las usan herramientas de análisis estático como `mypy` o `pyright`, y los editores para autocompletar y detectar errores antes de ejecutar el código. El módulo `typing` provee las construcciones para expresar tipos más complejos.

```python
def calcular_total(precios: list[float], descuento: float = 0.0) -> float:
    return sum(precios) * (1 - descuento)
```

Desde Python 3.9 se pueden usar las colecciones nativas como genéricos (`list[int]`, `dict[str, float]`, `tuple[int, ...]`) en lugar de `typing.List` o `typing.Dict`, que quedaron obsoletos.

## Uniones y valores opcionales

Desde Python 3.10 una unión se escribe con el operador `|`: `int | str` significa "un entero o un string". Un valor que puede faltar se anota como `str | None`, equivalente al antiguo `Optional[str]`. Ojo: `Optional` **no** significa que el parámetro sea opcional al llamar a la función, sino que puede valer `None`; para que sea opcional además necesita un valor por defecto.

```python
def buscar_usuario(email: str) -> dict | None:
    ...
```

Antes de usar el resultado, el chequeador obliga a descartar el caso `None` (por ejemplo con `if usuario is None: return`), lo que previene el clásico `AttributeError: 'NoneType' object has no attribute ...`.

## Literal y Annotated

`Literal["asc", "desc"]` restringe un valor a un conjunto fijo de constantes, ideal para parámetros que funcionan como opciones. `Annotated[int, "metadatos"]` agrega información extra a un tipo sin cambiarlo; librerías como Pydantic y FastAPI la usan para declarar validaciones (`Annotated[int, Field(gt=0)]`).

## TypedDict: la forma de un diccionario

`TypedDict` describe diccionarios con claves conocidas y un tipo para cada valor. En tiempo de ejecución siguen siendo diccionarios comunes, pero el chequeador detecta claves mal escritas o valores del tipo incorrecto:

```python
from typing import TypedDict, NotRequired

class Metadata(TypedDict):
    source: str
    page: int
    tags: NotRequired[list[str]]
```

Es una buena forma de fijar un **esquema** para datos que viajan como diccionarios, como la metadata de documentos en una base vectorial.

## Protocol: tipado estructural (duck typing estático)

Un `Protocol` define una interfaz por su **estructura** (qué métodos y atributos tiene) en lugar de por herencia. Cualquier clase que implemente esos métodos es compatible con el protocolo, aunque nunca herede de él. Es la versión estática del *duck typing*: "si camina como pato y hace cuac, es un pato".

```python
from typing import Protocol

class Embedder(Protocol):
    def embed(self, texto: str) -> list[float]: ...

class EmbedderOpenAI:          # no hereda de Embedder
    def embed(self, texto: str) -> list[float]:
        return [0.1, 0.2]

def indexar(e: Embedder, textos: list[str]) -> None: ...

indexar(EmbedderOpenAI(), ["hola"])   # válido para mypy
```

Con el decorador `@runtime_checkable` también se puede usar `isinstance(obj, Embedder)`, aunque solo comprueba que los métodos existan, no sus firmas. Para interfaces con herencia explícita y métodos abstractos obligatorios se usa `abc.ABC` con `@abstractmethod`.

## Genéricos y TypeVar

Los genéricos permiten escribir funciones y clases que funcionan con cualquier tipo, conservando la relación entre la entrada y la salida. Desde Python 3.12 (PEP 695) existe una sintaxis nativa:

```python
def primero[T](items: list[T]) -> T:
    return items[0]

class Caja[T]:
    def __init__(self, contenido: T) -> None:
        self.contenido = contenido

type Vector = list[float]   # alias de tipo (3.12+)
```

En versiones anteriores se declara `T = TypeVar("T")` y la clase hereda de `Generic[T]`. Un `TypeVar` puede tener un límite (`bound=`) para exigir que el tipo sea subclase de algo.

## Callable, overload y otros

- `Callable[[int, str], bool]`: una función que recibe un `int` y un `str` y devuelve `bool`. Hoy se recomienda importarlo desde `collections.abc`.
- `@overload`: declara varias firmas para una misma función cuando el tipo de retorno depende del tipo de los argumentos.
- `Any`: desactiva el chequeo para ese valor; conviene usarlo lo menos posible.
- `Self` (3.11+): el tipo de la propia clase, útil en métodos que devuelven `self`.
- `TYPE_CHECKING`: constante que vale `True` solo para el chequeador; permite importar tipos solo para anotaciones y evitar importaciones circulares.

## Verificar los tipos

Las anotaciones solo aportan valor si se verifican. Con `pip install mypy` y `mypy src/` se analiza el proyecto; la opción `--strict` activa todas las verificaciones. Para ver qué tipo infiere el chequeador en un punto del código se usa `reveal_type(expresion)`.
