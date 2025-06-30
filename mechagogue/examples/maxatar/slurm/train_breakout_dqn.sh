#!/bin/bash
#SBATCH -J maxatar_dqn_breakout          # job name
#SBATCH -p kempner_h100                  # partition (queue)
#SBATCH --account=kempner_sham_lab       # fairshare account
#SBATCH -N 1                             # number of nodes
#SBATCH --ntasks-per-node=1              # tasks per node
#SBATCH --cpus-per-task=16               # cpu cores per task, A100: 64 cores, H100: 96 cores
#SBATCH --gres=gpu:1                     # number of GPUs per node
#SBATCH --mem 128G                       # memory per node, H100: 1.5 TB, A100: 1 TB RAM
#SBATCH -t 00-03:00                      # time (D-HH:MM)
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

module load python/3.10.13-fasrc01
mamba activate mechagogue

cd ..

python breakout.py -e 1000000 --gif breakout_1M_epochs.gif
