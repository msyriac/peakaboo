#!/bin/bash 
#SBATCH -N 4 # node count 
#SBATCH -n  272 #
#########SBATCH -J recon
#SBATCH -t 1:00:00 #
#SBATCH --output=/work/02977/jialiu/PipelineJL/cmb-software/peakaboo/jobs/logs/recon_%j.out
#SBATCH --error=/work/02977/jialiu/PipelineJL/cmb-software/peakaboo/jobs/logs/recon_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jia@astro.princeton.edu 
#SBATCH -A TG-AST140041
#SBATCH -p normal #development 

peakapath=/work/02977/jialiu/PipelineJL/cmb-software/peakaboo
outdir=/scratch/02977/jialiu/peakaboo
inputdir=/scratch/02977/jialiu/lenstools_storage
fidudir=Om0.29997_As2.10000_mva0.00000_mvb0.00000_mvc0.00000_h0.70000_Ode0.69995

cd ${peakapath}


