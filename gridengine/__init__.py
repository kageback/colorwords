import gridengine.interactive as interactive


def start_job(gpu=0, timeout=0):
    interactive.start_job(gpu=gpu,timeout=timeout)