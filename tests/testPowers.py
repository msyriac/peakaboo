import numpy as np
from peakaboo.liuSims import LiuConvergence
import orphics.theory.cosmology as cosmo
import orphics.tools.cmb as cmb
import orphics.tools.io as io
import os,sys
import orphics.tools.stats as stats

def getPkzJia(pkzfile_root="/home/msyriac/data/jia/matterpower/mnv0.00000_om0.30000_As2.1000/*",h=0.7):
    import glob

    file_list = sorted(glob.glob(pkzfile_root))

    Pks = []
    zs = []
    for k,filen in enumerate(file_list[:66][::-1]):
        #print filen
        with open(filen,'r') as f:
            t = f.readline()
            a = f.readline()
            
            anum = float(a[a.find("=")+1:].strip())
            z = (1./anum)-1.
        ks,Pk = np.loadtxt(filen,unpack=True)
        ks = ks*1000.*h
        Pk = Pk/(1000**3.)/(h**3.)
        if k>0: assert np.all(np.isclose(ks,ksold))
        ksold = ks
        Pks.append(Pk)
        zs.append(z)

    Pks = np.array(Pks)
    zs = np.array(zs)
    
    # Ps = np.loadtxt(pkzfile)
    # H0 = 67.3
    # ks = Ps[:,0]*(H0/100.) # 1/Mpc
    # Ps = Ps[:,1:]*(2.*np.pi**2)/(ks.reshape([len(ks),1]))**3 # Mpc^3

    return ks,zs,Pks


out_dir = "./"
sim_pk_root = "/home/msyriac/data/jia/matterpower/mnv0.00000_om0.30000_As2.1000/powerspec_tot*"
nu = "massless"
lmax = 3000

# get some info
#L = LiuConvergence("/gpfs01/astro/workarea/msyriac/data/jiav2/"+nu+"/")
L = LiuConvergence("/home/msyriac/data/jia/"+nu+"/")
kappa = L.get_kappa(1)
modlmap = kappa.modlmap()
ellmin = np.sort(np.asarray(list(set((modlmap.ravel()).tolist())))).tolist()[3]
print "Lowest ell: ", ellmin
bin_edges = np.logspace(np.log10(ellmin),np.log10(lmax),10)
binner = stats.bin2D(modlmap,bin_edges)



pl = io.Plotter(scaleY='log')

# camb_cl_prediction
cambRoot = "data/jia_"+nu
theory = cmb.loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000)
clkk_camb_2d = theory.gCl("kk",modlmap)
cents,clkk_camb = binner.bin(clkk_camb_2d)
pl.add(cents,clkk_camb,label="cl camb")


# sim clkk power

cents_file,clkkraw = np.loadtxt("data/measured_power_mat_"+nu+".txt",unpack=True)
assert np.all(np.isclose(cents_file,cents))
clkk_sims = clkkraw/cents/(cents+1.)*2.*np.pi
pl.add(cents,clkk_sims,label="cl sims")


# orphics cl prediction by limber integral with same cosmo params as in camb_cl_prediction

defaultCosmology = {}
defaultCosmology['omch2'] = 0.12470
defaultCosmology['ombh2'] = 0.02230
defaultCosmology['H0'] = 70.0
defaultCosmology['ns'] = 0.97
defaultCosmology['As'] = 2.1e-9
defaultCosmology['mnu'] = 0.0
defaultCosmology['w0'] = -1.0

cc = cosmo.LimberCosmology(paramDict=defaultCosmology,lmax=3000,pickling=True,pkgrid_override=None)
ellrange = np.arange(2,lmax,1)
cc.generateCls(ellrange,autoOnly=True)
clkk_orphics_1d = cc.getCl("cmb","cmb")
from scipy.interpolate import interp1d
clkk_orphics_2d = interp1d(ellrange,clkk_orphics_1d,bounds_error=False,fill_value=np.nan)(modlmap)
cents,clkk_orphics = binner.bin(clkk_orphics_2d)
pl.add(cents,clkk_orphics,label="cl orphics+camb")


# pycamb matter power comparison


ks,zs,Pks = getPkzJia(h = defaultCosmology['H0']/100.)
print Pks.shape
Pk_orphics = cc.PK.P(zs,ks,grid=True)
print Pk_orphics.shape


# orphics cl prediction by limber integral with sim matter power

from scipy.interpolate import RectBivariateSpline
pkint = RectBivariateSpline(ks,zs,Pks.T,kx=3,ky=3)

cc_sim = cosmo.LimberCosmology(paramDict=defaultCosmology,lmax=3000,pickling=True,pkgrid_override=pkint)
cc_sim.generateCls(ellrange,autoOnly=True)
clkk_orphics_sim_1d = cc_sim.getCl("cmb","cmb")
from scipy.interpolate import interp1d
clkk_orphics_sim_2d = interp1d(ellrange,clkk_orphics_sim_1d,bounds_error=False,fill_value=np.nan)(modlmap)
cents,clkk_orphics_sim = binner.bin(clkk_orphics_sim_2d)
pl.add(cents,clkk_orphics,label="cl orphics+sim")





pl.legendOn(labsize=10,loc="lower left")
pl.done(out_dir+"clkk.png")


pl = io.Plotter()
perdiff_orphics = (clkk_orphics-clkk_camb)*100./clkk_camb
pl.add(cents,perdiff_orphics,label="orphics_camb vs. camb")
perdiff_orphics_sim = (clkk_orphics_sim-clkk_camb)*100./clkk_camb
pl.add(cents,perdiff_orphics_sim,label="orphics_sim vs. camb")
perdiff_sims = (clkk_sims-clkk_camb)*100./clkk_camb
pl.add(cents,perdiff_sims,label="sims vs. camb")
pl._ax.axhline(y=0.,ls="--",alpha=0.5,color="k")
pl.legendOn(labsize=10,loc="lower right")
pl.done(out_dir+"clkkdiff.png")


io.quickPlot2d(Pk_orphics,out_dir+"pkorphics.png")
io.quickPlot2d(Pks,out_dir+"pksim.png")

perdiff = np.nan_to_num((Pks-Pk_orphics)*100./Pk_orphics)
io.quickPlot2d(perdiff,out_dir+"pkperdiff.png")

pl = io.Plotter(scaleY='log',scaleX='log')
for i,z in enumerate(zs):
    simpk = Pks[i,:]
    orphpk = Pk_orphics[i,:]

    pl.add(ks,simpk)
    pl.add(ks,orphpk,ls="--",alpha=0.5)
pl._ax.axhline(y=0.,ls="--",alpha=0.5,color="k")
pl.legendOn(labsize=10,loc="lower right")
pl._ax.set_ylim(1e-6,1e7)
pl.done(out_dir+"pkcomp.png")



pl = io.Plotter(scaleX='log')
for i,z in enumerate(zs):
    simpk = Pks[i,:]
    orphpk = Pk_orphics[i,:]

    perdiff = np.nan_to_num((simpk-orphpk)*100./orphpk)
    pl.add(ks,perdiff)
pl._ax.axhline(y=0.,ls="--",alpha=0.5,color="k")
pl._ax.set_ylim(-20.,20.)
pl.legendOn(labsize=10,loc="lower right")
pl.done(out_dir+"pkdiff.png")
