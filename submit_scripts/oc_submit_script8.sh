#!/bin/bash
#PBS -S /bin/bash
#PBS -V
#PBS -W group_list=cosmo
#PBS -q high_pri
#PBS -J 1-1881
#PBS -l select=1:ncpus=1:mem=1GB
#PBS -l place=free:shared
#PBS -l walltime=5:00:00
#PBS -N LSST1_cov
#PBS -e /home/u17/timeifler/output/
#PBS -o /home/u17/timeifler/output/

module load gsl/2.1

cd $PBS_O_WORKDIR
/home/u17/timeifler/CosmoLike/LSSTC_obsstrat/./compute_covariances_fourier8 $PBS_ARRAY_INDEX >&/home/u17/timeifler/output/job_output_$PBS_ARRAY_INDEX.log




