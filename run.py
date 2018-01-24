import gridengine

#gridengine.start_job(timeout=20, gpu=1)


import gridengine.batch as ge

job = ge.Job()

job.run_python_script('run_exp.py', ge_gpu=1, batch_size=100, hidden=5)
#job.erase_job()

#job_id = ge.qsub('run.sh')
#print(job_id)