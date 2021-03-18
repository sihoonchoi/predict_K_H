#!/bin/bash
#PBS -N train-model
#PBS -q hive
#PBS -o output
#PBS -j oe
#PBS -l nodes=1:ppn=8
#PBS -l walltime=24:00:00
#PBS -l mem=32gb
#PBS -m abe
#PBS -M schoi420@gatech.edu

module load anaconda3

cd $PBS_O_WORKDIR

conda activate my_env

python pipeline.py set_1.csv set_2.csv
