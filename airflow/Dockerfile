# Imagem do Airflow com as dependências do treino já instaladas.
# Python 3.12 pra bater com o projeto (o código usa zip(strict=True), 3.10+).
FROM apache/airflow:2.10.4-python3.12

COPY requirements.txt /requirements.txt

# Reinstala o próprio apache-airflow na mesma versão junto com as deps: assim o
# pip não troca a versão do Airflow pra satisfazer alguma dependência nova.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
