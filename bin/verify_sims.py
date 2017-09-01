from liuSims import LiuConvergence
import orphics.tools.io as io
import os, sys
from orphics.theory.cosmology import Cosmology
from orphics.tools.stats import bin2D,getStats
import numpy as np
import orphics.analysis.flatMaps as fmaps
import orphics.tools.cmb as cmb

nu = "massless"
out_dir = os.environ['WWW']

ells = np.arange(2,3000,1)


jells,jcls = np.loadtxt("data/jiapower.txt",unpack=True)
#pells,pcls = np.loadtxt("data/ps_massless001.txt",unpack=True)
pells1,pcls1 = np.loadtxt("data/ps_massless1-9.txt",unpack=True)
pells2,pcls2 = np.loadtxt("data/ps_massive1-9.txt",unpack=True)
# print pells1
# print pells2
# sys.exit()

pl = io.Plotter(scaleY='log',scaleX='log')

cls = {}
cstats = {}
clkk = {}
for nu,ls,col in zip(['massless','massive'],['-','--'],['C0','C1']):

    cambRoot = "data/jia_"+nu
    theory = cmb.loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000)

    cls[nu] = []

    L = LiuConvergence("/gpfs01/astro/workarea/msyriac/data/jiav2/"+nu+"/")

    clkk[nu] = theory.gCl('kk',ells)

    for k,i in enumerate(range(1,200)):
        print i
        kappa = L.get_kappa(i)
        if k==0:
            modlmap = kappa.modlmap()
            ellmin = np.sort(np.asarray(list(set((modlmap.ravel()).tolist())))).tolist()[3]
            print "Lowest ell: ", ellmin
            bin_edges = np.logspace(np.log10(ellmin),np.log10(3000),10)

            binner = bin2D(modlmap,bin_edges)
            io.highResPlot2d(kappa,out_dir+"kappa1.png")

        k2d = fmaps.get_simple_power_enmap(kappa)
        cents, mkk = binner.bin(k2d)
        cls[nu].append(mkk)



        pl.add(cents,mkk*cents*(cents+1.)/2./np.pi,alpha=0.01,ls=ls)

    cstats[nu] = getStats(cls[nu])
    pl.addErr(cents,cstats[nu]['mean']*cents*(cents+1.)/2./np.pi,yerr=cstats[nu]['errmean']*cents*(cents+1.)/2./np.pi,marker="o",ls=ls,color=col)
    pl.add(ells,clkk[nu]*ells*(ells+1.)/2./np.pi,color="k",ls=ls)

    np.savetxt("camb_mat_"+nu+".txt",np.vstack((ells,clkk[nu]*ells*(ells+1.)/2./np.pi)).T)
    np.savetxt("measured_power_mat_"+nu+".txt",np.vstack((cents,cstats[nu]['mean']*cents*(cents+1.)/2./np.pi)).T)


#pl.add(jells,jcls,color="C2",lw=3)
#pl.add(pells1,pcls1,color="C3",lw=2,label="jmassless")
#pl.add(pells2,pcls2,color="C4",lw=2,label="jmassive")
    
pl._ax.set_ylim(2e-4,3e-3)
pl._ax.set_xlim(100,4000)
pl.legendOn(loc="lower right",labsize=10)
pl.done(out_dir+"clkk.png")


pl = io.Plotter(scaleX='log',labelX="$L$",labelY="$(m-m_0)/m_0$")
perdiff = (cstats["massive"]['mean']-cstats["massless"]['mean'])/cstats["massless"]['mean']
thdiff = (clkk["massive"]-clkk["massless"])/clkk["massless"]
jdiff = (pcls2-pcls1)/pcls1
#pl.add(pells1,jdiff,ls="-.",label="jia")
pl.add(ells,thdiff,label="camb")
pl.add(cents,perdiff,ls="--",label="mat")
pl.legendOn(labsize=10)
pl._ax.set_xlim(100,4000)
pl.done(out_dir+"pdiff.png")
