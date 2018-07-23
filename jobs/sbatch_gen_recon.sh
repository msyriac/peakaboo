peakapath=/work/02977/jialiu/PipelineJL/cmb-software/peakaboo/
outdir=/scratch/02977/jialiu/peakaboo/
inputdir=/scratch/02977/jialiu/lenstools_storage/


while read cosmodir
do
fn=recon/recon_${cosmodir:0:9}.sh
cp template_recon.sh ${fn}
echo ${cosmodir}
echo "#ibrun -n 136 -o 0 python ${peakapath}/bin/recon.py ${cosmodir}/1024b512 output_tt_3000_s4 reconstruction_liu_TT_3000 experiment_s4 TT default_1d_bin &" >>${fn}
echo " " >> ${fn}

echo "ibrun -n 136 -o 136 python ${peakapath}/bin/recon.py ${cosmodir}/1024b512 output_eb_5000_s4 reconstruction_liu_EB_5000 experiment_s4 EB default_1d_bin &" >> ${fn}
echo " " >> ${fn}
echo "wait" >> ${fn}
echo " " >> ${fn}
done < ${peakapath}/cosmo_dir.ls
