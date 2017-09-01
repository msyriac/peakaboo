from enlib import enmap,utils,lensing,powspec
import copy
import flipper.liteMap as lm
import flipper.fft as fft
import os, sys
import orphics.analysis.flatMaps as fmaps
import numpy as np
import orphics.tools.io as io
import orphics.tools.cmb as cmb
from szar.counts import ClusterCosmology
from orphics.tools.stats import bin2D,getStats
from scipy.interpolate import interp1d
import flipperPol.liteMapPol as lpol

size_deg = 3.0
px = 2.0
TCMB = 2.7255e6
N = 10


out_dir = os.environ['WWW']

# === TEMPLATE MAP ===
arc = size_deg*60.
hwidth = arc/2.
deg = utils.degree
arcmin =  utils.arcmin
shape, wcs = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")
Ny,Nx = shape
shape = (3,)+shape

class template:
    def __init__(self):
        pass
    def copy(self):
        return copy.copy(self)


T = template()
T.Nx = Nx
T.Ny = Ny
T.pixScaleX = px*np.pi/180./60.
T.pixScaleY = px*np.pi/180./60.

def twrap(array,temp):
    tempC = temp.copy()
    tempC.data = array.copy()
    return tempC

lxMap,lyMap,modLMap,angMap,lx,ly = fmaps.getFTAttributesFromLiteMap(T)

#ps = powspec.read_spectrum("/astro/astronfs01/workarea/msyriac/data/sigurd_sims/planck15_lenspotentialCls.dat")
ps = powspec.read_spectrum("/astro/astronfs01/workarea/msyriac/data/sigurd_sims/cl_lensed.dat")
theory = cmb.loadTheorySpectraFromCAMB("/astro/astronfs01/workarea/msyriac/data/sigurd_sims/planck15")
ellfine = np.arange(2,8000,1)


avg_ute = 0.


flipperpol = False


for i in range(N):
    print i
    map = enmap.rand_map(shape, wcs, ps)/TCMB
    # mt,mq,mu = lpol.simPolMapsFromEandB(T,ellfine,theory.uCl('TT',ellfine),theory.uCl('EE',ellfine),theory.uCl('TE',ellfine))
    # map[0] = mt.data.copy()
    # map[1] = mq.data.copy()
    # map[2] = mu.data.copy()
    

    if flipperpol:
        fot,foe,fob = fmaps.TQUtoFourierTEB(map[0],map[1],map[2],modLMap,angMap)
        teb = []
        teb.append(fft.ifft(fot,axes=[-2,-1],normalize=True).real)
        teb.append(fft.ifft(foe,axes=[-2,-1],normalize=True).real)
        teb.append(fft.ifft(fob,axes=[-2,-1],normalize=True).real)
    else:
        teb = enmap.ifft(enmap.map2harm(map)).real

    uT = teb[0]
    uE = teb[1]
    ones = uT.copy()*0.+1.

    avg_ute += fmaps.get_simple_power(twrap(uT,T),ones,twrap(uE,T),ones)/N
    


cmb_ellbins = np.arange(110,8000,20)
cmb_binner = bin2D(modLMap,cmb_ellbins)
ls,ute = cmb_binner.bin(avg_ute)
    
pl = io.Plotter()
pl.add(ellfine,ellfine**2.*theory.uCl('TE',ellfine),ls="--")
pl.add(ls,ute*ls**2.,ls="none",marker="o",label="u",alpha=0.3,color="C0")
pl._ax.set_xlim(2.,3000.)
pl.legendOn(loc="upper right")
pl.done(out_dir+"clte.png")
    
    
