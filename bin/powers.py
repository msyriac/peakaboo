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
import argparse


# Parse command line
parser = argparse.ArgumentParser(description='Run lensing sims pipe.')
parser.add_argument("InpDir", type=str,help='Input Directory Name (not path, that\'s specified in ini)')
parser.add_argument("OutDir", type=str,help='Output Directory Name')
parser.add_argument("bin_section", type=str,help='bin_section')
parser.add_argument("-N", "--nmax",     type=int,  default=1000,help="Limit to nmax sims.")
parser.add_argument("-G", "--galaxies",     type=str,
                    default=None,help="Comma separated list of galaxy redshifts.")
args = parser.parse_args()



# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

inp_dir = args.InpDir
out_dir = args.OutDir


galzs = [int(x) for x in args.galaxies.split(',')]

PathConfig = io.load_path_config()
result_dir = PathConfig.get("paths","output_data")+inp_dir+"/"+out_dir+"/"

lcents,Nlkk = np.loadtxt(result_dir+"nlkk.txt",unpack=True)

file_root = lambda sim_id: result_dir+"kappa_"+str(sim_id).zfill(4)+".fits"

Ntot = args.nmax
num_each,each_tasks = mpi_distribute(Ntot,numcores)
mpibox = MPIStats(comm,num_each,tag_start=333)
my_tasks = each_tasks[rank]
if rank==0: print( "At most "+ str(max(num_each)) + " tasks...")

map_root = PathConfig.get("paths","input_data")
LC = ls.LiuConvergence(root_dir=map_root+inp_dir+"/")

Config = io.config_from_file("input/recon.ini")
lbin_edges = io.bin_edges_from_config(Config,args.bin_section)



for k,i in enumerate(my_tasks):

    recon = enmap.read_map(file_root(i))


    if k==0:
        shape,wcs = recon.shape,recon.wcs
        fc = enmap.FourierCalc(shape,wcs)
        lbinner = bin2D(recon.modlmap(),lbin_edges)
        
    
    input_k = enmap.ndmap(resample.resample_fft(LC.get_kappa(i+1,z=1100),shape),wcs)
    

    p2drcic,krc,kic = fc.power2d(recon,input_k)
    cents, prcic = lbinner.bin(p2drcic)
    assert np.all(np.isclose(cents,lcents))
    p2dicic  = fc.f2power(kic,kic)
    cents, picic = lbinner.bin(p2dicic)
    p2drcrc  = fc.f2power(krc,krc)
    cents, prcrc = lbinner.bin(p2drcrc)


    for z in galzs:
        galkappa = enmap.ndmap(resample.resample_fft(LC.get_kappa(i+1,z=z),shape),wcs)
        p2dicig,kig = fc.f1power(galkappa,kic)
        cents, picig = lbinner.bin(p2dicig)
        p2drcig = fc.f2power(krc,kig)
        cents, prcig = lbinner.bin(p2drcig)
        

    mpibox.add_to_stats("icig",picig)
    mpibox.add_to_stats("rcig",prcig)
    mpibox.add_to_stats("icic",picic)
    mpibox.add_to_stats("rcic",prcic)
    mpibox.add_to_stats("rcrc",prcrc)

    if rank==0 and (k+1)%1==0: print( "Rank 0 done with "+str(k+1)+ " / "+str( len(my_tasks))+ " tasks.")
    
    
mpibox.get_stats()

if rank==0:
    rcrc = mpibox.stats["rcrc"]["mean"]
    icig = mpibox.stats["icig"]["mean"]
    rcig = mpibox.stats["rcig"]["mean"]
    rcig_err = mpibox.stats["rcig"]["err"]
    inputk = mpibox.stats["icic"]["mean"]
    cross = mpibox.stats["rcic"]["mean"]
    cross_err = mpibox.stats["rcic"]["err"]

    print (inputk)
    print (cross)
    
    pdiff = (cross-inputk)/inputk
    
    pl = io.Plotter(labelX="$\\ell$",labelY="$\\Delta C_{\ell}/C_{\ell}$")
    pl.add(cents,pdiff,color="k")
    pl.hline()
    #pl._ax.set_ylim(-2,1)
    pl._ax.set_xlim(lbin_edges[0],lbin_edges[1-])
    pl.done(io.dout_dir+"pdiff.png")


    pl = io.Plotter(labelX="$\\ell$",labelY="$C_{\ell}$",scaleY='log')
    pl.add(cents,inputk,color="k")
    pl.add(lcents,Nlkk,ls="--")
    pl.add(lcents,Nlkk+inputk,ls="-")
    pl.add(cents,rcrc,marker="o",ls="none")
    pl.addErr(cents,cross,yerr=cross_err,marker="o")
    pl._ax.set_xlim(lbin_edges[0],lbin_edges[1-])
    pl.done(io.dout_dir+"clkk.png")


    
    pl = io.Plotter(labelX="$\\ell$",labelY="$\\ell C_{\ell}$")
    pl.add(cents,icig*cents,color="k")
    pl.addErr(cents,rcig*cents,yerr=rcig_err*cents,marker="o")
    pl._ax.set_xlim(lbin_edges[0],lbin_edges[1-])
    pl.done(io.dout_dir+"galcross.png")
