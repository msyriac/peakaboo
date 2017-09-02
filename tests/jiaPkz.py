import numpy as np
import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
from orphics.theory.cosmology import LimberCosmology
from scipy.interpolate import interp1d
import sys,os

pkzfile = "../orphics/data/May21_cdm_1.0_cut_ibarrier_iconc.dat"
pkzfile2 = "../orphics/data/May21_fdm_1.0_cut_ibarrier_iconc.dat"


def perdiff(a,b):
    return (a-b)*100./b


def getPkz(pkzfile):

    with open(pkzfile,'r') as f:
        t = f.readline()
        zs =  np.array([float(x) for x in t.split()[1:]])

        
    Ps = np.loadtxt(pkzfile)
    H0 = 67.3
    ks = Ps[:,0]*(H0/100.) # 1/Mpc
    Ps = Ps[:,1:]*(2.*np.pi**2)/(ks.reshape([len(ks),1]))**3 # Mpc^3


    return ks,zs,Ps

def getPkzJia(pkzfile_root):
    import glob

    file_list = sorted(glob.glob(pkzfile_root))

    Pks = []
    for k,filen in enumerate(file_list[:66][::-1]):
        #print filen
        with open(filen,'r') as f:
            t = f.readline()
            a = f.readline()
            
            anum = float(a[a.find("=")+1:].strip())
            z = (1./anum)-1.
            #print filen,a,z
            #zs =  np.array([float(x) for x in t.split()[1:]])
        ks,Pk = np.loadtxt(filen,unpack=True)
        ks = ks*1000.
        Pk = Pk/(1000**3.)
        if k>0: assert np.all(np.isclose(ks,ksold))
        ksold = ks
        #print z
        pl.add(ks,Pk)

        
    # Ps = np.loadtxt(pkzfile)
    # H0 = 67.3
    # ks = Ps[:,0]*(H0/100.) # 1/Mpc
    # Ps = Ps[:,1:]*(2.*np.pi**2)/(ks.reshape([len(ks),1]))**3 # Mpc^3


    # return ks,zs,Ps


pl = io.Plotter(scaleY='log',scaleX='log')
getPkzJia("/home/msyriac/data/jia/matterpower/mnv0.00000_om0.30000_As2.1000/*")


ks,zs,Pkz = getPkz(pkzfile)

print zs.shape
print Pkz.shape

for i in range(Pkz.shape[1]):
   if zs[i]>0 and zs[i]<47:
       pl.add(ks,Pkz[:,i],ls="-",alpha=0.5)


pl.done("pks.png")
    
sys.exit()

from scipy.interpolate import RectBivariateSpline
pkint = RectBivariateSpline(ks,zs,Pkz,kx=3,ky=3)



ks2,zs2,Pkz2 = getPkz(pkzfile2)


kseval = ks
zseval = zs


io.quickPlot2d(Pkz,"pkz.png")
io.quickPlot2d(pkint(kseval,zseval),"pkzint.png")

lc = LimberCosmology(lmax=8000,pickling=True,kmax=400.,pkgrid_override=pkint)

from alhazen.shear import dndz

step = 0.01
zedges = np.arange(0.,3.0+step,step)
zcents = (zedges[1:]+zedges[:-1])/2.
dndzs = dndz(zcents)


pkz_camb = lc.PK.P(zseval, kseval, grid=True).T
io.quickPlot2d(pkz_camb,"pkz_camb.png")

pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(kseval,pkz_camb[:,0])
pl.add(ks,Pkz[:,0],ls="--")
pl.add(ks2,Pkz2[:,0],ls="--")
pl.done("pk0.png")


cdmfile = "data/May21_matter2lens_WF_CDM_cut_ibarrier_iconc_fCls.csv"
fdmfile = "data/May21_matter2lens_WF_FDM_1.0_cut_ibarrier_iconc_fCls.csv"

lcdm,ckkcdm = np.loadtxt(cdmfile,unpack=True)
lfdm,ckkfdm = np.loadtxt(fdmfile,unpack=True)


lc.addNz("s",zedges,dndzs,bias=None,magbias=None,numzIntegral=300)
lc.addStepNz("g",0.4,0.7,bias=2,magbias=None,numzIntegral=300)


ellrange = np.arange(100,8000,1)
lc.generateCls(ellrange,autoOnly=False,zmin=0.)

clkk = lc.getCl("cmb","cmb")
clss = lc.getCl("s","s")
clsk = lc.getCl("cmb","s")
clsg = lc.getCl("s","g")
clgk = lc.getCl("cmb","g")
clgg = lc.getCl("g","g")


pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ellrange,clkk,color="C0",label="kk")
pl.add(ellrange,clss,color="C1",label="ss")
pl.add(ellrange,clsk,color="C2",label="sk")
pl.add(ellrange,clgg,color="C3",label="gg")
pl.add(ellrange,clsg,color="C4",label="sg")
pl.add(ellrange,clgk,color="C5",label="gk")



       

pl.add(lcdm,ckkcdm,ls="none",marker="o",markersize=1,color="C0",alpha=0.1)
pl.add(lfdm,ckkfdm,ls="none",marker="x",markersize=1,color="C0",alpha=0.1)


