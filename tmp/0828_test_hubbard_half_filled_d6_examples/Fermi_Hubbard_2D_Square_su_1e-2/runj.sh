#!/bin/bash

#SBATCH -J Hub_D6_SU
#SBATCH -p batch 
#SBATCH -N 1 --ntasks-per-node=64
#SBATCH --time=2-0:0:0
#SBATCH --mem=224G
##SBATCH -d afterok:287
#SBATCH -o out.%j
#SBATCH -e err.%j

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export FI_PROVIDER=verbs
export UCX_NET_DEVICES=mlx5_0:1

cd $SLURM_SUBMIT_DIR
scontrol show hostname $SLURM_NODELIST > host.txt

JULIA_DEPOT_PATH="/gpfs/home/jgkong/software/MyProject/Grassmann_Bohr" julia $1 > out.txt
echo 'done'
