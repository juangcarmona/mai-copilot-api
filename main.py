import os
import sys
import uvicorn
from pathlib import Path

# Añadir la carpeta `src` al path de Python
project_root = Path(__file__).parent.resolve()  # Ruta de la raíz del proyecto
src_path = project_root / "src"  # Ruta a la carpeta src
sys.path.append(str(src_path))  # Añadir src al PYTHONPATH

# Configurar el generador predeterminado
os.environ.setdefault("DEFAULT_GENERATOR", "tinystarcoder")

if __name__ == "__main__":
    uvicorn.run(
        "mai.api:app",  # Especificamos el módulo y la app
        host="0.0.0.0",
        port=8000,
        reload=True
    )
