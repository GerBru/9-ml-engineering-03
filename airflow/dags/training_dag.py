"""DAG de treino do classificador de laudos médicos (Tech Challenge Fase 3).

    ingest_data -> validate_data -> train_model -> evaluate_model -> deploy_model

Este arquivo só orquestra: a lógica de cada task está em training/dag_tasks.py.
O modelo só é publicado em models/ se passar no quality gate do evaluate_model.
"""

import logging
from datetime import timedelta

import pendulum
from airflow.decorators import dag, task

logger = logging.getLogger(__name__)


def alert_failure(context):
    """Chamada quando uma task falha. Em produção, mandaria pro Slack/e-mail."""
    ti = context["task_instance"]
    logger.error(
        "🚨 ALERTA: task %s falhou na DAG %s (run %s). Log: %s",
        ti.task_id,
        ti.dag_id,
        ti.run_id,
        ti.log_url,
    )


default_args = {
    "owner": "german",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
    "on_failure_callback": alert_failure,
}


# O import de training.dag_tasks fica dentro de cada task, e não no topo do
# arquivo: ele carrega pandas, scikit-learn e scipy, e o Airflow relê este
# arquivo a cada 30 s e no início de cada task. Com o import no topo, cada
# leitura levava ~20 s no Windows e estourava o limite de 30 s do Airflow.


@task(task_id="ingest_data")
def ingest_data(run_id: str) -> str:
    from training import dag_tasks

    return dag_tasks.ingest_data(run_id)


@task(task_id="validate_data")
def validate_data(staging_dir: str) -> str:
    from training import dag_tasks

    return dag_tasks.validate_staged_data(staging_dir)


@task(task_id="train_model")
def train_model(staging_dir: str) -> str:
    from training import dag_tasks

    return dag_tasks.train_staged_model(staging_dir)


# Reprovar no quality gate não é falha transitória: tentar de novo dá o mesmo
# resultado, então essa task não tem retry.
@task(task_id="evaluate_model", retries=0)
def evaluate_model(staging_dir: str) -> dict:
    from training import dag_tasks

    return dag_tasks.evaluate_staged_model(staging_dir)


@task(task_id="deploy_model")
def deploy_model(resultado: dict) -> str:
    from training import dag_tasks

    return dag_tasks.deploy_model(resultado)


@dag(
    dag_id="medical_triage_training",
    description="Treina, avalia e publica o classificador de laudos médicos",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 10, 1, tz="America/Sao_Paulo"),
    schedule=None,  # só manual; pra retreino periódico seria ex.: "0 2 * * 1"
    catchup=False,
    tags=["tech-challenge", "ml", "treino"],
    doc_md=__doc__,
)
def medical_triage_training():
    staging_dir = ingest_data(run_id="{{ run_id }}")
    staging_dir = validate_data(staging_dir)
    staging_dir = train_model(staging_dir)
    resultado = evaluate_model(staging_dir)
    deploy_model(resultado)


medical_triage_training()
