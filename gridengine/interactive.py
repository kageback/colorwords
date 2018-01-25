import subprocess
import os
import signal
import time
import argparse
import socket

pool_path = 'pool' + '.' + socket.gethostname()

def start_job(gpu=0, timeout=0):
    # Check if GPU already assigned
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('CUDA_VISIBLE_DEVICES externally set to ' + os.environ['CUDA_VISIBLE_DEVICES'])
        return

    #Check the pool for a idle job
    start_new_job = True
    if os.path.isfile(pool_path):
        file = open(pool_path, 'r+')
        if file.readline() == '':
            file.seek(0)
            file.write(str(os.getpid()))
            start_new_job = False
        file.close()

    # Start new grid engine job using qsub
    if start_new_job:
        self_path = os.path.dirname(os.path.realpath(__file__))
        cmd = 'qsub -cwd -o interactive.log -e interactive.log -b y -l gpu=' + str(gpu) \
              + ' python3 -u ' + self_path + '/interactive.py --pid ' + str(os.getpid()) + ' --timeout ' + str(timeout)
        try:
            subprocess.call(cmd.split())
        except OSError as e:
            print('Failed to run qsub! Continues to run locally instead, i.e. outside grid engine.')
            return

    # Wait for job to start
    envfile = '.env.' + str(os.getpid()) + '.' + socket.gethostname()
    while not os.path.isfile(envfile):
        time.sleep(1)

    _load_env(envfile)
    os.remove(envfile)

def _load_env(envfile):
    file = open(envfile, 'r')
    os.environ['CUDA_VISIBLE_DEVICES'] = file.readline()
    print('CUDA_VISIBLE_DEVICES set to ' + os.environ['CUDA_VISIBLE_DEVICES'])
    file.close()

# ============ Code below is run by grid engine job (queue job) ====================

def _save_env(envfile):
    file = open(envfile, 'w')
    file.write(os.environ.get('CUDA_VISIBLE_DEVICES', '-1'))
    file.close()


def _exit_gracefully(signum, frame):
    print('Queue job received external termination signal (maybe from qdel).')

    if state == 'running':
        print('Terminating batch job before proceeding to terminate queue job.')
        os.kill(args.pid, signal.SIGTERM)
    elif state == 'pooled':
        print('Removing job from pool')
        os.remove(pool_path)

    print('===================================================')
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool for easy management of GE on Titan')
    parser.add_argument('--pid', type=int, default=-1,
                        help='Pid of the main process running the experiment.')
    parser.add_argument('--timeout', type=int, default=0,
                        help='How long a job will be kept alive after process dies to be used for next process.')
    args = parser.parse_args()

    envfile = '.env.' + str(args.pid) + '.' + socket.gethostname()
    _save_env(envfile)

    signal.signal(signal.SIGINT, _exit_gracefully)
    signal.signal(signal.SIGTERM, _exit_gracefully)

    state = 'running'
    print(state)
    while True:
        if state == 'running':
            if not os.path.exists('/proc/' + str(args.pid)):
                if args.timeout == 0:
                    break
                elif os.path.isfile(pool_path):
                    print('Already a job in the pool. Terminating queue job.')
                    break
                else:
                    open(pool_path, 'w').close()
                    pool_start_time = time.time()
                    state = 'pooled'
                    print(state)
        elif state == 'pooled':
            file = open(pool_path, 'r')
            pid = file.readline()
            file.close()
            if not pid == '':
                os.remove(pool_path)
                args.pid = int(pid)
                envfile = '.env.' + str(args.pid) + '.' + socket.gethostname()
                _save_env(envfile)
                state = 'running'
                print(state)
            elif (time.time()-pool_start_time) > args.timeout:
                print('pool timed out.')
                os.remove(pool_path)
                break

        time.sleep(1)

    print('Batch job has been terminated. Exiting Queue job...')
    print('===================================================')
