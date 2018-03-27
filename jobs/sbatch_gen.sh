peakapath=/work/02977/jialiu/PipelineJL/cmb-software/peakaboo
outdir=/scratch/02977/jialiu/peakaboo
inputdir=/scratch/02977/jialiu/lenstools_storage
fidudir=Om0.29997_As2.10000_mva0.00000_mvb0.00000_mvc0.00000_h0.70000_Ode0.69995

cd ${peakapath}/jobs

while read cosmodir
do 
cp template.sh ${cosmodir}_stats.sh

echo ${cosmodir}

echo "ibrun -n 136 -o 0 python -W ignore bin/powers.py   ${cosmodir}/1024b512 output_eb_5000_s4 default_1d_bin default_hist_bin coarse_hist_bin_cmb,coarse_hist_bin_gal ${fidudir}/1024b512 -G 0.5,1,1.5,2,2.5 -x 1,2,5 -y 0.5,1,2,5 -N 1000 &

ibrun -n 136 -o 136 python -W ignore bin/powers.py   ${cosmodir}/1024b512 output_tt_3000_s4 default_1d_bin default_hist_bin coarse_hist_bin_cmb,coarse_hist_bin_gal ${fidudir}/1024b512 -G 0.5,1,1.5,2,2.5 -x 1,2,5 -y 0.5,1,2,5 -N 1000 &

wait" >> ${cosmodir}_stats.sh
done < ${peakapath}/cosmo_dir.ls
