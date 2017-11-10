from __future__ import print_function
import numpy as np
from peakaboo import utils as ls
import orphics.tools.io as io
import os,sys
import orphics.analysis.flatMaps as fmaps
from orphics.tools.mpi import mpi_distribute, MPIStats, MPI
import orphics.analysis.flatMaps as fmaps
from enlib import enmap, resample
from orphics.tools.stats import bin2D

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

inp_dir = "cmb/massless"
out_dir = "test"

PathConfig = io.load_path_config()
result_dir = PathConfig.get("paths","output_data")+inp_dir+"/"+out_dir+"/"

file_root = lambda sim_id: result_dir+"kappa_"+str(sim_id).zfill(4)+".fits"

Ntot = 1000
num_each,each_tasks = mpi_distribute(Ntot,numcores)
mpibox = MPIStats(comm,num_each,tag_start=333)
my_tasks = each_tasks[rank]

map_root = PathConfig.get("paths","input_data")
LC = ls.LiuConvergence(root_dir=map_root+inp_dir+"/",zstr="1100.00")

lbin_edges = np.arange(200,3000,100)



for k,i in enumerate(my_tasks):

    recon = enmap.read_map(file_root(i))


    if k==0:
        shape,wcs = recon.shape,recon.wcs
        fc = enmap.FourierCalc(shape,wcs)
        lbinner = bin2D(recon.modlmap(),lbin_edges)
    
    input_k = enmap.ndmap(resample.resample_fft(LC.get_kappa(i+1),shape),wcs)
    

    p2dcross,krecon,kinput = fc.power2d(recon,input_k)
    cents, pcross = lbinner.bin(p2dcross)
    p2dauto  = fc.f2power(kinput,kinput)
    cents, pauto_input = lbinner.bin(p2dauto)

    mpibox.add_to_stats("input",pauto_input)
    mpibox.add_to_stats("cross",pcross)

    if rank==0 and (k+1)%1==0: print( "Rank 0 done with "+str(k+1)+ " / "+str( len(my_tasks))+ " tasks.")
    
    
mpibox.get_stats()

if rank==0:
    inputk = mpibox.stats["input"]["mean"]
    cross = mpibox.stats["cross"]["mean"]

    print (inputk)
    print (cross)
    
    pdiff = (cross-inputk)/inputk
    
    pl = io.Plotter(labelX="$\\ell$",labelY="$\\Delta C_{\ell}/C_{\ell}$")
    pl.add(cents,pdiff,color="k")
    pl.hline()
    #pl._ax.set_ylim(-2,1)
    pl._ax.set_xlim(500,3000)
    pl.done(io.dout_dir+"pdiff.png")
