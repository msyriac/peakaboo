import numpy as np
from peakaboo import liuSims as ls
import orphics.tools.io as io
import os,sys
import orphics.analysis.flatMaps as fmaps
from mpi4py import MPI
from orphics.analysis.pipeline import mpi_distribute, MPIStats
import orphics.analysis.flatMaps as fmaps
from enlib import enmap, resample

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


out_dir = os.environ['WWW']+"peakaboo/"

#file_root = "/gpfs01/astro/workarea/msyriac/data/sims/jia/output/jia_recon_"
file_root = lambda mass,ftype,x,ext: "/gpfs01/astro/workarea/msyriac/data/sims/jia/output/"+mass+"_"+ftype+"_experiment_simple_"+str(x).zfill(9)+"."+ext

Ntot = 500
num_each,each_tasks = mpi_distribute(Ntot,numcores)
mpibox = MPIStats(comm,num_each,tag_start=333)
my_tasks = each_tasks[rank]

LCmassless = ls.LiuConvergence(root_dir="/gpfs01/astro/workarea/msyriac/data/sims/jia/cmb/massless/",zstr="1100.00")
LCmassive = ls.LiuConvergence(root_dir="/gpfs01/astro/workarea/msyriac/data/sims/jia/cmb/massive/",zstr="1100.00")

lbin_edges = np.arange(200,3000,100)



for k,i in enumerate(my_tasks):
    #massive = enmap.read_map(file_root+"massive_"+str(i).zfill(9)+".fits")
    #massless = enmap.read_map(file_root+"massless_"+str(i).zfill(9)+".fits")

    massive = enmap.read_map(file_root("massive","kappa_recon",i,"fits"))
    massless = enmap.read_map(file_root("massless","kappa_recon",i,"fits"))

    if k==0:
        qpower = fmaps.QuickPower(massive.modlmap(),lbin_edges)
    
    massive_input = LCmassive.get_kappa(i+1)
    massive_input = enmap.ndmap(resample.resample_fft(massive_input,massive.shape),massive.wcs)
    massless_input = LCmassless.get_kappa(i+1)
    massless_input = enmap.ndmap(resample.resample_fft(massless_input,massless.shape),massless.wcs)

    print massive.shape
    print massive_input.shape

    cents, pauto_massive = qpower.calc(massive)
    cents, pauto_massless = qpower.calc(massless)
    cents, pcross_massive = qpower.calc(massive,massive_input)
    cents, pcross_massless = qpower.calc(massless,massless_input)

    cents, pauto_massive_input = qpower.calc(massive_input)
    cents, pauto_massless_input = qpower.calc(massless_input)

    lcents,massive_rkk = np.loadtxt(file_root("massive","auto_n0_subbed",i,"fits"),unpack=True)
    lcents,massless_rkk = np.loadtxt(file_root("massless","auto_n0_subbed",i,"fits"),unpack=True)

    mpibox.add_to_stats("massiveAutoN0",massive_rkk)
    mpibox.add_to_stats("masslessAutoN0",massless_rkk)

    
    mpibox.add_to_stats("massiveAuto",pauto_massive)
    mpibox.add_to_stats("masslessAuto",pauto_massless)
    mpibox.add_to_stats("masslessCross",pcross_massless)
    mpibox.add_to_stats("massiveCross",pcross_massive)
    mpibox.add_to_stats("massiveInput",pauto_massive_input)
    mpibox.add_to_stats("masslessInput",pauto_massless_input)
    
    
    print rank,i
    
mpibox.get_stats()

if rank==0:
    rm = mpibox.stats["massiveAutoN0"]
    rm0 = mpibox.stats["masslessAutoN0"]
    mauto = mpibox.stats["massiveAuto"]
    m0auto = mpibox.stats["masslessAuto"]
    m0cross = mpibox.stats["masslessCross"]
    mcross = mpibox.stats["massiveCross"]
    mauto_input = mpibox.stats["massiveInput"]
    m0auto_input = mpibox.stats["masslessInput"]

    def camb_pred(nu):
        import orphics.tools.cmb as cmb
        # camb_cl_prediction
        cambRoot = "data/jia_"+nu
        theory = cmb.loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000)
        ellrange = np.arange(2,3000,1)
        clkk_camb = theory.gCl("kk",ellrange)
        return ellrange,clkk_camb


    ellrange,clkk_camb0 = camb_pred("massless")
    
    pl = io.Plotter(scaleY='log',labelX="$\\ell$",labelY="$C_{\ell}$")
    pl.addErr(lcents,rm0['mean'],yerr=rm0['errmean'],marker="o")
    pl.addErr(cents,m0cross['mean'],yerr=m0cross['errmean'],marker="^")
    pl.add(cents,m0auto_input['mean'],marker="x",ls="none")
    pl.add(ellrange,clkk_camb0,label="cl camb",color="k")
    pl._ax.set_ylim(1.e-9,1.e-6)
    pl.done(out_dir+"massless.png")

    ellrange,clkk_camb = camb_pred("massive")

    pl = io.Plotter(scaleY='log',labelX="$\\ell$",labelY="$C_{\ell}$")
    pl.addErr(lcents,rm['mean'],yerr=rm['errmean'],marker="o")
    pl.addErr(cents,mcross['mean'],yerr=mcross['errmean'],marker="^")
    pl.add(cents,mauto_input['mean'],marker="x",ls="none")
    pl.add(ellrange,clkk_camb,label="cl camb",color="k")
    pl._ax.set_ylim(1.e-9,1.e-6)
    pl.done(out_dir+"massive.png")

    pdiff = (clkk_camb-clkk_camb0)*100./clkk_camb0
    
    pl = io.Plotter(labelX="$\\ell$",labelY="$100\\Delta C_{\ell}/C_{\ell}$")
    pl.add(lcents,(rm['mean']-rm0['mean'])*100./rm0['mean'],marker="o",ls="none")
    pl.add(cents,(mauto_input['mean']-m0auto_input['mean'])*100./m0auto_input['mean'],marker="x",ls="none")
    pl.add(ellrange,pdiff,label="cl camb",color="k")
    pl.hline()
    #pl._ax.set_ylim(-2,1)
    pl._ax.set_xlim(500,3000)
    pl.done(out_dir+"mnudiff.png")
