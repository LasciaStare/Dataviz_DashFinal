# Usar imagen base oficial de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar archivos de requisitos primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos de la aplicación
COPY . .

# Exponer el puerto que usa la aplicación
EXPOSE 8050

# Comando para ejecutar la aplicación con gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8050", "--timeout", "120", "--workers", "2", "dashboard:server"]