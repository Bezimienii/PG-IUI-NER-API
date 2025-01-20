from apscheduler.schedulers.background import BackgroundScheduler
from filelock import FileLock
import multiprocessing

from ..config import settings
from ..database.context_manager import Session
from ..database.models import ProcessId, AIModel
from psutil import pid_exists
import psutil
from os import getpid

from ..model.training import execute_training


def get_lock() -> FileLock:
    return FileLock(settings.SYNC_DIR + "/counter.lock")

#It might have happened that
#there are processes that have died
#between death and restart of API
def get_not_running_processes(processes: [ProcessId]) -> [ProcessId]:
    not_running_processes = []
    for process in processes:
        pid = process.pid
        if not pid_exists(int(pid)):
            not_running_processes.append(process)
            continue
        p = psutil.Process(int(pid))
        name = p.name()
        if name != process.name:
            not_running_processes.append(process)
    return not_running_processes


def check_and_clean():
    lock = get_lock()
    with lock:
        with Session() as session:
            processes = session.query(ProcessId).all()
            not_running_processes = get_not_running_processes(processes)
            print(not_running_processes)
            for process in not_running_processes:
                session.delete(process)
            session.commit()


def write_subprocess():
    pid = getpid()
    process = psutil.Process(pid)

    lock = get_lock()
    with lock:
        processId = ProcessId(pid=str(pid), name=process.name())
        with Session() as session:
            session.add(processId)
            session.commit()

def delete_subprocess():
    pid = str(getpid())
    lock = get_lock()

    with lock:
        with Session() as session:
            processId = session.query(ProcessId).filter_by(pid=pid).first()
            session.delete(processId)
            session.commit()

def sortFunc(obj):
    return obj.id

def job_callback(max_subprocesses):
    def job():
        lock = get_lock()
        with lock:
            with Session() as session:
                processes = session.query(ProcessId).all()
                available_subprocesses = max_subprocesses - len(processes)
                if available_subprocesses > 0:
                    models = session.query(AIModel).filter_by(is_training=False,is_trained=False).all()
                    models.sort(key=sortFunc)
                    chosenModels = models if len(models) < available_subprocesses else models[0:2]
                    for model in chosenModels:
                        p = multiprocessing.Process(target=execute_training, args=(model.id,))  # process independent of parent
                        p.start()
    return job


def start_job(scheduler: BackgroundScheduler):
    job = job_callback(settings.MAX_PROCESSES)
    check_and_clean()
    job()
    scheduler.add_job(job, 'interval', minutes=2)

