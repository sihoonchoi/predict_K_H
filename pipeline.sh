#!/bin/bash
#PBS -N train-model
#PBS -q hive
#PBS -o output
#PBS -j oe
#PBS -l nodes=1:ppn=8
#PBS -l walltime=12:00:00
#PBS -l mem=32gb
#PBS -m abe
#PBS -M schoi420@gatech.edu

cd $PBS_O_WORKDIR

module load anaconda3

python pipeline.py set_1.csv set_2.csv