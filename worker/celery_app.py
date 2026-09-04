"""
Celery application instance.

Redis is used as BOTH the message broker (holds pending task messages)
and the result backend (stores task state + return value), as required
by the task spec.
"""
import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_track_started=True,          # so status can report STARTED, not just PENDING
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,              # keep results for 1 hour
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Allow tests to run tasks synchronously in-process when set.
    task_always_eager=os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true",
    task_eager_propagates=True,
)
