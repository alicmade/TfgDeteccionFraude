#solo una iamgen, que tenga pyspark
#FROM apache/spark-py
#USER root
FROM openjdk:11-slim

# Instala Python y pip
RUN apt-get update && apt-get install -y python3 python3-pip python3-distutils && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Rutas correctas
ENV JAVA_HOME=/usr/local/openjdk-11
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYSPARK_PYTHON=python3

# Instalar tus dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#COPY credenciales.json /app/credenciales.json

# Copiar tu código
COPY . /app
WORKDIR /app

CMD ["python", "main.py"]


