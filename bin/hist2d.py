import numpy as np
from peakaboo import liuSims as ls
import orphics.tools.io as io
import os,sys
import orphics.analysis.flatMaps as fmaps
from mpi4py import MPI
import argparse
from orphics.analysis.pipeline import mpi_distribute, MPIStats

parser = argparse.ArgumentParser(description='Do all the things.')
parser.add_argument("-N", "--nsim",     type=int,  default=None)
args = parser.parse_args()

Nsims = args.nsim


# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    



# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Nsims,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)

def get_cents(edges):
    return (edges[1:]+edges[:-1])/2.


# What am I doing?
my_tasks = each_tasks[rank]

out_dir = os.environ['WWW']+"peakaboo/"

dw = 0.001
bin_edges = np.arange(-0.3,0.3+dw,dw)+dw
bincents = get_cents(bin_edges)


dw2d = 0.01
bin_edges2d = np.arange(-0.3,0.3+dw2d,dw2d)+dw2d

lbin_edges = np.arange(400,3000,200)

for mnu in ["massive","massless"]:

    LCc = ls.LiuConvergence(root_dir="/gpfs01/astro/workarea/msyriac/data/sims/jia/cmb/"+mnu+"/",zstr="1100.00")
    LCg1 = ls.LiuConvergence(root_dir="/gpfs01/astro/workarea/msyriac/data/sims/jia/gal10/"+mnu+"/",zstr="1.00")
    LCg15 = ls.LiuConvergence(root_dir="/gpfs01/astro/workarea/msyriac/data/sims/jia/gal15/"+mnu+"/",zstr="1.50")
    LCg2 = ls.LiuConvergence(root_dir="/gpfs01/astro/workarea/msyriac/data/sims/jia/gal20/"+mnu+"/",zstr="2.00")
    qpower = fmaps.QuickPower(LCc.modlmap,lbin_edges)


    for k,index in enumerate(my_tasks):

        cmb = LCc.get_kappa(index+1)
        # if rank==0 and k==0: io.highResPlot2d(cmb,out_dir+"cmb_"+mnu+".png")
        gal1 = LCg1.get_kappa(index+1)
        # if rank==0 and k==0: io.highResPlot2d(gal1,out_dir+"gal1_"+mnu+".png")
        gal15 = LCg15.get_kappa(index+1)
        # if rank==0 and k==0: io.highResPlot2d(gal15,out_dir+"gal15_"+mnu+".png")
        gal2 = LCg2.get_kappa(index+1)
        # if rank==0 and k==0: io.highResPlot2d(gal2,out_dir+"gal2_"+mnu+".png")



        hist1d_cmb, bin_edges = np.histogram(cmb.ravel(),bin_edges,density=True)
        hist1d_gal1, bin_edges = np.histogram(gal1.ravel(),bin_edges,density=True)
        hist1d_gal15, bin_edges = np.histogram(gal15.ravel(),bin_edges,density=True)
        hist1d_gal2, bin_edges = np.histogram(gal2.ravel(),bin_edges,density=True)


        mpibox.add_to_stack("cmb_1d_"+mnu,hist1d_cmb)
        mpibox.add_to_stack("gal1_1d_"+mnu,hist1d_gal1)
        mpibox.add_to_stack("gal15_1d_"+mnu,hist1d_gal15)
        mpibox.add_to_stack("gal2_1d_"+mnu,hist1d_gal2)
        



        hist2d1, x_edges,y_edges = np.histogram2d(cmb.ravel(),gal1.ravel(),bin_edges2d,normed=True)
        hist2d15, x_edges,y_edges = np.histogram2d(cmb.ravel(),gal15.ravel(),bin_edges2d,normed=True)
        hist2d2, x_edges,y_edges = np.histogram2d(cmb.ravel(),gal2.ravel(),bin_edges2d,normed=True)

        mpibox.add_to_stack("cgal1_2d_"+mnu,hist2d1)
        mpibox.add_to_stack("cgal15_2d_"+mnu,hist2d15)
        mpibox.add_to_stack("cgal2_2d_"+mnu,hist2d2)


        cents, pauto_cmb = qpower.calc(cmb)
        cents, pauto_gal1 = qpower.calc(gal1)
        cents, pcross1 = qpower.calc(cmb,gal1)
        cents, pauto_gal15 = qpower.calc(gal15)
        cents, pcross15 = qpower.calc(cmb,gal15)
        cents, pauto_gal2 = qpower.calc(gal2)
        cents, pcross2 = qpower.calc(cmb,gal2)


        mpibox.add_to_stats("auto_cmb_"+mnu,pauto_cmb)
        mpibox.add_to_stats("auto_gal1_"+mnu,pauto_gal1)
        mpibox.add_to_stats("cross_gal1_"+mnu,pcross1)
        mpibox.add_to_stats("auto_gal15_"+mnu,pauto_gal15)
        mpibox.add_to_stats("cross_gal15_"+mnu,pcross15)
        mpibox.add_to_stats("auto_gal2_"+mnu,pauto_gal2)
        mpibox.add_to_stats("cross_gal2_"+mnu,pcross2)

        if rank==0: print mnu,index+1

mpibox.get_stacks()
mpibox.get_stats()
        
if rank==0:

    import matplotlib.pyplot as plt

    pd = lambda x,y: (x-y)*100./y
    pd1d = lambda x: (mpibox.stacks[x+"massive"]-mpibox.stacks[x+"massless"])*100./mpibox.stacks[x+"massless"]
    
    pl = io.Plotter()
    for mnu,ls in zip(["massive","massless"],["-","--"]):
        pl.add(bincents,mpibox.stacks["cmb_1d_"+mnu],ls=ls)
        pl.add(bincents,mpibox.stacks["gal1_1d_"+mnu],ls=ls)
        pl.add(bincents,mpibox.stacks["gal15_1d_"+mnu],ls=ls)
        pl.add(bincents,mpibox.stacks["gal2_1d_"+mnu],ls=ls)
        plt.gca().set_prop_cycle(None)
    pl.done(out_dir+"hist1d.png")
    pl = io.Plotter()
    pl.add(bincents,pd1d("cmb_1d_"),ls=ls)
    pl.add(bincents,pd1d("gal1_1d_"),ls=ls)
    pl.add(bincents,pd1d("gal15_1d_"),ls=ls)
    pl.add(bincents,pd1d("gal2_1d_"),ls=ls)
    pl._ax.set_ylim(-20,10)
    pl.hline()
    pl.done(out_dir+"hist1diff.png")

    
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.mlab as mlab

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    x = get_cents(x_edges)
    y = get_cents(y_edges)
    X, Y = np.meshgrid(x, y)
    
    plt.figure()
    for mnu,ls in zip(["massive","massless"],["-","--"]):
        for lab in ["cgal1_2d_"+mnu,"cgal15_2d_"+mnu,"cgal2_2d_"+mnu]:
            Z = mpibox.stacks[lab]
            #CS = plt.contour(X, Y, Z,levels=[2,10,30],linestyles=[ls])
            CS = plt.contour(X, Y, Z,levels=[10],linestyles=[ls])
    #plt.clabel(CS, inline=1, fontsize=10)
    plt.savefig(out_dir+"hist2d.png")


    plt.clf()




    from orphics.theory.cosmology import LimberCosmology

    defaultCosmology = {}
    defaultCosmology['omch2'] = 0.12470
    defaultCosmology['ombh2'] = 0.02230
    defaultCosmology['H0'] = 70.
    defaultCosmology['ns'] = 0.97
    defaultCosmology['As'] = 2.1e-9
    defaultCosmology['mnu'] = 0.0
    defaultCosmology['w0'] = -1.0


    
    lcmassless = LimberCosmology(defaultCosmology,lmax=3000,pickling=True,nonlinear=False)
    lcmassless.addDeltaNz("z1",1.0)
    lcmassless.addDeltaNz("z15",1.5)
    lcmassless.addDeltaNz("z2",2.0)
    ellrange = np.arange(400,3000,1)
    lcmassless.generateCls(ellrange)
    clkk0 = lcmassless.getCl("cmb","cmb")


    defaultCosmology = {}
    defaultCosmology['omch2'] = 0.12362
    defaultCosmology['ombh2'] = 0.02230
    defaultCosmology['H0'] = 70.
    defaultCosmology['ns'] = 0.97
    defaultCosmology['As'] = 2.1e-9
    defaultCosmology['mnu'] = 0.1
    defaultCosmology['w0'] = -1.0


    lcmassive = LimberCosmology(defaultCosmology,lmax=3000,pickling=True,nonlinear=False)
    lcmassive.addDeltaNz("z1",1.0)
    lcmassive.addDeltaNz("z15",1.5)
    lcmassive.addDeltaNz("z2",2.0)
    ellrange = np.arange(400,3000,1)
    lcmassive.generateCls(ellrange)
    clkk = lcmassive.getCl("cmb","cmb")

    

    pl = io.Plotter(labelX="$\\ell$",labelY="$\\ell C_{\\ell}$",scaleY='log')
    for mnu,ls in zip(["massive","massless"],["--","-"]):
        if mnu=="massless":
            mark ="^"
        else:
            mark="o"
        acstats = mpibox.stats["auto_cmb_"+mnu]
        pl.addErr(cents,acstats['mean']*cents,yerr=acstats['errmean']*cents,ls="none",marker="o" if mnu=="massive" else "^",color="C0",markersize=4)
        atheory = lcmassless.getCl("cmb","cmb")
        pl.add(ellrange,ellrange*atheory,color="C0")
        for k,z in enumerate(["1","15","2"]):
            atheory = lcmassless.getCl("z"+z,"z"+z)
            ctheory = lcmassless.getCl("z"+z,"cmb")
            agstats = mpibox.stats["auto_gal"+z+"_"+mnu]
            cstats = mpibox.stats["cross_gal"+z+"_"+mnu]
            if mnu=="massless":
                pl.add(ellrange,ellrange*atheory,color="C"+str(k+1))
                pl.add(ellrange,ellrange*ctheory,color="C"+str(k+2))
                
            pl.addErr(cents,agstats['mean']*cents,yerr=agstats['errmean']*cents,ls="none",marker=mark,color="C"+str(k+1),markersize=4)
            pl.addErr(cents,cstats['mean']*cents,yerr=cstats['errmean']*cents,ls="none",marker=mark,color="C"+str(k+2),markersize=4)
        plt.gca().set_prop_cycle(None)
    pl.hline()
    pl._ax.set_ylim(1.e-7,3.e-5)
    pl.legendOn(loc="lower right",labsize=12)
    pl.done(out_dir+"cpower.png")

    pl = io.Plotter(labelX="$\\ell$",labelY="$\\Delta C_{\\ell}/C_{\\ell}$")
    acstats = mpibox.stats["auto_cmb_massive"]
    acstats0 = mpibox.stats["auto_cmb_massless"]
    atheory = lcmassive.getCl("cmb","cmb")
    atheory0 = lcmassless.getCl("cmb","cmb")
    perdiff = lambda x,y: (x-y)/y
    pl.add(ellrange,perdiff(atheory,atheory0),color="C0")
    pl.add(cents,perdiff(acstats['mean'],acstats0['mean']),ls="none",marker=mark,color="C0",markersize=4)
    for k,(z,col) in enumerate(zip(["1","15","2"][:1],["C1","C2","C3"][:1])):
        atheory = lcmassive.getCl("z"+z,"z"+z)
        atheory0 = lcmassless.getCl("z"+z,"z"+z)
        pl.add(ellrange,perdiff(atheory,atheory0),color="C1")
        ctheory = lcmassive.getCl("z"+z,"cmb")
        ctheory0 = lcmassless.getCl("z"+z,"cmb")
        pl.add(ellrange,perdiff(ctheory,ctheory0),ls="--",color="C2")
        agstats = mpibox.stats["auto_gal"+z+"_massive"]
        cstats = mpibox.stats["cross_gal"+z+"_massive"]
        agstats0 = mpibox.stats["auto_gal"+z+"_massless"]
        cstats0 = mpibox.stats["cross_gal"+z+"_massless"]
                
        pl.add(cents,perdiff(agstats['mean'],agstats0['mean']),ls="none",marker=mark,color="C1",markersize=4)
        pl.add(cents,perdiff(cstats['mean'],cstats0['mean']),ls="none",marker=mark,color="C2",markersize=4)
        #plt.gca().set_prop_cycle(None)
    pl.hline()
    #pl._ax.set_ylim(1.e-7,3.e-5)
    pl.legendOn(loc="lower right",labsize=12)
    pl.done(out_dir+"cpowerdiff.png")

    

    # pl = io.Plotter(labelX="$\\ell$",labelY="$100\\Delta C_{\\ell} / C_{\\ell}$",scaleY='log')
    #     for k,z in enumerate(["cmb","1","15","2"]):
            
    #         acstats = mpibox.stats["auto_cmb_"+mnu]
    #         agstats = mpibox.stats["auto_gal"+z+"_"+mnu]
    #         cstats = mpibox.stats["cross_gal"+z+"_"+mnu]
            
    #         atheory = lcmassless.getCl("z"+z,"z"+z)
    #         ctheory = lcmassless.getCl("z"+z,"cmb")
    #         agstats = mpibox.stats["auto_gal"+z+"_"+mnu]
    #         cstats = mpibox.stats["cross_gal"+z+"_"+mnu]
    #         if mnu=="massless":
    #             pl.add(ellrange,ellrange*atheory,color="C"+str(k+1))
    #             pl.add(ellrange,ellrange*ctheory,color="C"+str(k+2))
    #         pl.addErr(cents,agstats['mean']*cents,yerr=agstats['errmean']*cents,ls="none",marker="o" if mnu=="massive" else "^",color="C"+str(k+1))
    #         pl.addErr(cents,cstats['mean']*cents,yerr=cstats['errmean']*cents,ls="none",marker="o" if mnu=="massive" else "^",color="C"+str(k+2))
    #     plt.gca().set_prop_cycle(None)
    # pl.hline()
    # pl._ax.set_ylim(1.e-7,3.e-5)
    # pl.legendOn(loc="lower right",labsize=12)
    # pl.done(out_dir+"cpowerdiff.png")
