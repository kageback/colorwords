#!/bin/sh
#
# Startup script to allocate GPU devices.
#
# Kota Yamaguchi 2015 <kyamagu@vision.is.tohoku.ac.jp>

# Check if the environment file is writable.
ENV_FILE=$SGE_JOB_SPOOL_DIR/environment
if [ ! -f $ENV_FILE -o ! -w $ENV_FILE ]
then
  echo "ERROR: Environment file ("$SGE_JOB_SPOOL_DIR"/environment) is not writable!"
  exit 1
fi

# Write environment file location to home folder (used by qrsh users)
# USER_ENV_FILE=$SGE_O_HOME/sge_env_path
# echo $ENV_FILE > $USER_ENV_FILE

# Query how many gpus to allocate.
NGPUS=$(qstat -j $JOB_ID | \
        sed -n "s/hard resource_list:.*gpu=\([[:digit:]]\+\).*/\1/p")
if [ -z $NGPUS ]
then
  echo CUDA_VISIBLE_DEVICES=-1 >> $ENV_FILE
  echo "No GPU requested! Add -l gpu=1 to your qsub command to request one gpu if you need GPU power."
  exit 0
fi
if [ $NGPUS -le 0 ]
then
  echo CUDA_VISIBLE_DEVICES=-1 >> $ENV_FILE
  echo "No GPU requested! Add -l gpu=1 to your qsub command to request one gpu if you need GPU power."
  exit 0
fi
NGPUS=$(expr $NGPUS \* ${NSLOTS=1})

# Allocate and lock GPUs.
SGE_GPU=""
i=0
device_ids=$(nvidia-smi -L | cut -f1 -d":" | cut -f2 -d" " | xargs shuf -e)
for device_id in $device_ids
do
  lockfile=/tmp/lock-gpu$device_id
  if mkdir $lockfile
  then
    SGE_GPU="$SGE_GPU $device_id"
    i=$(expr $i + 1)
    if [ $i -ge $NGPUS ]
    then
      break
    fi
  fi
done

if [ $i -lt $NGPUS ]
then
  echo "ERROR: Only reserved $i of $NGPUS requested devices."
  exit 1
fi

# Set the environment.
echo SGE_GPU="$(echo $SGE_GPU | sed -e 's/^ //' | sed -e 's/ /,/g')" >> $ENV_FILE
echo "Allocated GPU(s) with id(s): "$SGE_GPU
echo CUDA_VISIBLE_DEVICES="$(echo $SGE_GPU | sed -e 's/^ //' | sed -e 's/ /,/g')" >> $ENV_FILE
exit 0
