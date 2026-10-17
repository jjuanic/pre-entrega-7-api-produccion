---
title: pathlib — Rutas del sistema de archivos orientadas a objetos
category: archivos
library: pathlib
tags: rutas, archivos, directorios, glob, os.path
---

# pathlib — Rutas del sistema de archivos orientadas a objetos

`pathlib` representa las rutas de archivos y directorios como **objetos** en lugar de strings. Reemplaza con ventaja a la mayoría de las funciones de `os.path` y funciona igual en Windows, Linux y macOS: la clase `Path` se instancia automáticamente como `WindowsPath` o `PosixPath` según el sistema operativo.

## Construir rutas

```python
from pathlib import Path

base = Path(__file__).resolve().parent   # carpeta del script actual
datos = base / "data" / "clientes.csv"    # el operador / une partes de la ruta
print(Path.cwd())    # directorio de trabajo actual
print(Path.home())   # carpeta del usuario
```

El operador `/` reemplaza a `os.path.join` y se encarga del separador correcto de cada sistema operativo. Usar `Path(__file__).resolve().parent` como base evita el clásico error de rutas relativas que funcionan o no según desde qué carpeta se ejecuta el script.

## Partes de una ruta

Dada `p = Path("/proyectos/app/informe.final.pdf")`:

| Atributo | Valor |
|---|---|
| `p.name` | `informe.final.pdf` |
| `p.stem` | `informe.final` |
| `p.suffix` | `.pdf` |
| `p.suffixes` | `['.final', '.pdf']` |
| `p.parent` | `/proyectos/app` |
| `p.parts` | `('/', 'proyectos', 'app', 'informe.final.pdf')` |

Para cambiar partes se usan `p.with_suffix(".txt")`, `p.with_name("otro.pdf")` o `p.with_stem("resumen")` (3.9+), que devuelven rutas nuevas: los objetos `Path` son inmutables.

## Leer y escribir archivos

Para archivos chicos, `Path` tiene atajos que abren, leen o escriben y cierran el archivo en una sola llamada:

```python
config = Path("config.json")
texto = config.read_text(encoding="utf-8")
config.write_text(texto.upper(), encoding="utf-8")

imagen = Path("logo.png").read_bytes()
```

Conviene pasar siempre `encoding="utf-8"` explícitamente: si no, se usa la codificación por defecto del sistema, que en Windows suele ser `cp1252` y provoca `UnicodeDecodeError` con acentos o emojis. Para archivos grandes o lectura línea por línea se usa `with p.open(encoding="utf-8") as f:`.

## Crear, mover y borrar

- `p.mkdir(parents=True, exist_ok=True)`: crea el directorio y todos los intermedios que falten, sin error si ya existe.
- `p.touch()`: crea un archivo vacío (o actualiza su fecha de modificación).
- `p.rename(destino)` / `p.replace(destino)`: mueve o renombra; `replace` sobrescribe el destino si existe.
- `p.unlink(missing_ok=True)`: borra un archivo.
- `p.rmdir()`: borra un directorio **vacío**. Para borrar un árbol completo se usa `shutil.rmtree(p)`.
- `p.exists()`, `p.is_file()`, `p.is_dir()`: consultas sobre el estado de la ruta.
- `p.stat().st_size` y `p.stat().st_mtime`: tamaño en bytes y fecha de última modificación.

## Recorrer directorios y buscar archivos

`p.iterdir()` lista el contenido inmediato de un directorio (sin recursión). Para buscar por patrón se usa `glob`:

```python
carpeta = Path("src")
scripts = list(carpeta.glob("*.py"))          # solo el primer nivel
todos = sorted(carpeta.rglob("*.py"))         # recursivo: todos los .py del árbol
tests = list(carpeta.glob("**/test_*.py"))   # equivalente con el comodín **
```

`rglob(patron)` es lo mismo que `glob("**/" + patron)`: recorre recursivamente todos los subdirectorios. Ambos devuelven generadores, por eso es común envolverlos en `list()` o `sorted()`. Desde Python 3.12 existe además `Path.walk()`, equivalente orientado a objetos de `os.walk`.

## Rutas absolutas, relativas y comparaciones

- `p.resolve()`: devuelve la ruta absoluta, resolviendo `..` y enlaces simbólicos.
- `p.absolute()`: la hace absoluta sin resolver enlaces.
- `p.relative_to(base)`: calcula la ruta relativa a `base` (lanza `ValueError` si `p` no está dentro de `base`); `p.is_relative_to(base)` lo verifica sin lanzar error.
- `p.expanduser()`: expande `~` a la carpeta del usuario.
- `p.match("*.py")` y `p.full_match("src/**/*.py")` (3.13+): comprueban si la ruta coincide con un patrón.

## Equivalencias con os.path

| os / os.path | pathlib |
|---|---|
| `os.path.join(a, b)` | `Path(a) / b` |
| `os.path.exists(p)` | `Path(p).exists()` |
| `os.path.basename(p)` | `Path(p).name` |
| `os.path.splitext(p)[1]` | `Path(p).suffix` |
| `os.makedirs(p, exist_ok=True)` | `Path(p).mkdir(parents=True, exist_ok=True)` |
| `os.listdir(p)` | `Path(p).iterdir()` |
| `glob.glob("**/*.py", recursive=True)` | `Path().rglob("*.py")` |

Las funciones de la librería estándar que reciben rutas (`open`, `shutil.copy`, `json.load` vía `open`) aceptan objetos `Path` directamente, porque implementan el protocolo `os.PathLike`. Si una librería externa exige un string, se convierte con `str(p)`.
