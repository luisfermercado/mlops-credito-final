# Imagen base Python 3.11 ligera
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar dependencias PRIMERO (aprovecha caché de Docker en rebuilds)
# Si solo cambia main.py, Docker salta esta capa y el rebuild es instantáneo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar solo el código de la API
# El modelo NO está aquí — vive en Databricks
COPY main.py .

# Puerto donde escucha la API
EXPOSE 8000

# Arrancar el servidor al iniciar el contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
