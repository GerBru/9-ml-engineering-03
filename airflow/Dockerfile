# Imagem do Airflow com as dependências do treino já instaladas.
# Python 3.12 pra bater com o projeto (o código usa zip(strict=True), 3.10+).
FROM apache/airflow:2.10.4-python3.12

# O grafo ONNX tem um nó StringNormalizer (o lowercase do TF-IDF) que exige o
# locale en_US.UTF-8, gravado no próprio grafo pelo skl2onnx. A imagem base só
# traz C e C.utf8, e sem este locale o check_parity do deploy_model quebra
# dentro do container (funciona fora dele, onde o locale existe).
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends locales \
 && sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen \
 && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements.txt /requirements.txt

# Reinstala o próprio apache-airflow na mesma versão junto com as deps: assim o
# pip não troca a versão do Airflow pra satisfazer alguma dependência nova.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
