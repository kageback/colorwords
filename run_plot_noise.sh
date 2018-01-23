#!/bin/bash

#EXP_NAME="test_GE"
#LOG_PATH='save/'$EXP_NAME'.log'
#qsub -N $EXP_NAME -cwd -l gpu=1 -b y -o $LOG_PATH -e $LOG_PATH python3 -u train.py --name $EXP_NAME --max_len 100 --min_freq 3 --covar_reg_rate 0.1

qsub -cwd -l gpu=1 -b y python3 -u plot_commcost_noise.py
