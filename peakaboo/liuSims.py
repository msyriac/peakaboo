from enlib import enmap,utils,lensing,powspec
import copy
from alhazen.halos import NFWkappa
from alhazen.lensTools import alphaMaker
import flipper.liteMap as lm
import flipper.fft as fft
import astropy.io.fits as fits
import os, sys
import orphics.analysis.flatMaps as fmaps
import numpy as np
from ConfigParser import SafeConfigParser 
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
import orphics.tools.io as io
import orphics.tools.cmb as cmb
from szar.counts import ClusterCosmology
# from alhazen.quadraticEstimator import Estimator
# from scipy.interpolate import interp1d


class LiuConvergence(object):

    def __init__(self,root_dir="/gpfs01/astro/workarea/msyriac/data/jia/"):


        self.root = root_dir
        self.kappa_file = lambda i :root_dir+"WLconv_z1100.00_"+str(i).zfill(4)+"r.fits"

        

        
        size_deg = 3.5
        Npix = 2048.
        px = size_deg*60./Npix

        self.shape, self.wcs = enmap.get_enmap_patch(size_deg*60.,px,proj="CAR",pol=False)
        self.lxmap,self.lymap,self.modlmap,self.angmap,self.lx,self.ly = fmaps.get_ft_attributes_enmap(self.shape,self.wcs)

        
    def get_kappa(self,index):

        
        my_map = fits.open(self.kappa_file(index))[0]
        my_map = my_map.data
        
        try:
            assert my_map.shape == self.shape
        except:
            print my_map.shape
            print self.shape
            sys.exit()

        self.low_pass_ell = 10000
        retmap = enmap.ndmap(my_map,self.wcs)
        retmap = enmap.ndmap(fmaps.filter_map(retmap,retmap.copy()*0.+1.,self.modlmap,lowPass=self.low_pass_ell),self.wcs)
            
        return retmap
        
# # === TEMPLATE MAP ===
# px = px
# arc = size_deg*60.
# hwidth = arc/2.
# deg = utils.degree
# arcmin =  utils.arcmin
# shape, wcs = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")
# shape = (3,)+shape
# pxDown = 0.5
# shapeDown, wcsDown = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=pxDown*arcmin, proj="car")
# shapeDown = (3,)+shape
# thetaMap = enmap.posmap(shape, wcs)
# thetaMap = np.sum(thetaMap**2,0)**0.5
# thetaMapDown = enmap.posmap(shapeDown, wcsDown)
# thetaMapDown = np.sum(thetaMapDown**2,0)**0.5


# # === COSMOLOGY ===
# cosmologyName = 'params' # from ini file
# iniFile = "../szar/input/pipeline.ini"
# Config = SafeConfigParser()
# Config.optionxform=str
# Config.read(iniFile)
# lmax = 8000
# cosmoDict = dictFromSection(Config,cosmologyName)
# constDict = dictFromSection(Config,'constants')
# cc = ClusterCosmology(cosmoDict,constDict,lmax,pickling=True)
# #theory = cc.theory
# TCMB = 2.7255e6

# # === CMB POWER SPECTRUM ===      
# ps = powspec.read_spectrum("/astro/astronfs01/workarea/msyriac/data/sigurd_sims/planck15_unlensed.dat")
# theory = cmb.loadTheorySpectraFromCAMB("/astro/astronfs01/workarea/msyriac/data/sigurd_sims/planck15",unlensedEqualsLensed=True)

# # === DEFLECTION MAP ===
# a = alphaMaker(thetaMap)

# def twrap(array,temp):
#     tempC = temp.copy()
#     tempC.data = array.copy()
#     return tempC

# gradCut = 10000
# cmbellmin = 100
# cmbellmax = 5000
# kellmin = 100
# kellmax = 6000

# beam = 1.0
# beamTemplate = cmb.gauss_beam(modLMap,beam)
# noiseT = 1.0
# noiseP = 1.0
# nT,nP,nP = fmaps.whiteNoise2D([noiseT,noiseP,noiseP],beam,modLMap,TCMB = TCMB)
# fMaskCMB = fmaps.fourierMask(lx,ly,modLMap,lmin=cmbellmin,lmax=cmbellmax)
# fMask = fmaps.fourierMask(lx,ly,modLMap,lmin=kellmin,lmax=kellmax)
# qest = Estimator(T,
#                  theory,
#                  theorySpectraForNorm=None,
#                  noiseX2dTEB=[nT,nP,nP],
#                  noiseY2dTEB=[nT,nP,nP],
#                  fmaskX2dTEB=[fMaskCMB]*3,
#                  fmaskY2dTEB=[fMaskCMB]*3,
#                  fmaskKappa=fMask,
#                  doCurl=False,
#                  TOnly=False,
#                  halo=True,
#                  gradCut=gradCut,verbose=True,
#                  loadPickledNormAndFilters=None,
#                  savePickledNormAndFilters=None)

# polcombs = ["TT","EB"]
# ellfine = np.arange(2,8000,1)

# ggen = fmaps.GRFGen(T,ellfine,theory.gCl("kk",ellfine))
# ggenT = fmaps.GRFGen(T,power2d=np.nan_to_num(nT*beamTemplate**2.))
# ggenP = fmaps.GRFGen(T,power2d=np.nan_to_num(nP*beamTemplate**2.))


# for i in range(1,N+1):

#     my_map = fits.open(input_kappa_file(i))

#     img = my_map[0].data
#     T.data = img
#     #kappaMap = thetaMap*0.+fmaps.filter_map(img,img.copy()*0.+1.,modLMap,lowPass=low_pass_ell)
#     kappaMap = thetaMap*0.+ggen.getMap(ellfine[-1])
#     ones = kappaMap*0.+1.
#     inp_kappa_lm = T.copy()
#     inp_kappa_lm.data = kappaMap
#     inp_power_2d = fmaps.get_simple_power(inp_kappa_lm,ones)


#     alpha = a.kappaToAlpha(kappaMap,test=False)
#     pos = thetaMap.posmap() + alpha
#     pix = thetaMap.sky2pix(pos, safe=False)

    
#     io.highResPlot2d(kappaMap, out_dir+"origkappa.png")

#     min_ell = np.sort(modLMap.ravel())[1]
#     print "min ell: ", min_ell
#     dell = 300
#     ellbins = np.arange(min_ell,kellmax,dell)
#     binner = bin2D(modLMap,ellbins)
#     ls,input_power = binner.bin(inp_power_2d)
    

#     Nsims = 200
#     kappa_avg = {}
#     cross_powers = {}
#     autos = {}
#     npows = {}
#     avg_utt = 0.
#     avg_ltt = 0.
#     avg_ute = 0.
#     avg_lte = 0.
#     avg_uee = 0.
#     avg_ubb = 0.
#     avg_lee = 0.
#     avg_lbb = 0.
#     cmb_ellbins = np.arange(110,8000,20)
#     cmb_binner = bin2D(modLMap,cmb_ellbins)

#     for polcomb in polcombs:
#         cross_powers[polcomb] = []
#         autos[polcomb] = []
#         kappa_avg[polcomb] = 0.
#     for k in range(Nsims):
#         map = enmap.rand_map(shape, wcs, ps)/TCMB
#         if k==0:
#             io.highResPlot2d(map[0], out_dir+"unlensedI.png")
#             io.highResPlot2d(map[1], out_dir+"unlensedQ.png")
#             io.highResPlot2d(map[2], out_dir+"unlensedU.png")

#         teb = enmap.ifft(enmap.map2harm(map)).real
#         uT = teb[0]
#         uE = teb[1]
#         uB = teb[2]
#         avg_utt += fmaps.get_simple_power(twrap(uT,T),ones)/Nsims
#         avg_uee += fmaps.get_simple_power(twrap(uE,T),ones)/Nsims
#         avg_ubb += fmaps.get_simple_power(twrap(uB,T),ones)/Nsims
#         avg_ute += fmaps.get_simple_power(twrap(uT,T),ones,twrap(uE,T),ones)/Nsims

#         if k==0:
#             io.highResPlot2d(uE, out_dir+"unlensedE.png")
#             io.highResPlot2d(uB, out_dir+"unlensedB.png")

            
#         lensedTQU = lensing.displace_map(map, pix,order=5)+0.j
#         if k==0:
#             io.highResPlot2d(lensedTQU[0], out_dir+"lensedI.png")
#             io.highResPlot2d(lensedTQU[1], out_dir+"lensedQ.png")
#             io.highResPlot2d(lensedTQU[2], out_dir+"lensedU.png")
#         # if k==0:
#         #     diff = lensedTQU-map
#         #     io.highResPlot2d(diff[0], out_dir+"diffI.png")
#         #     io.highResPlot2d(diff[1], out_dir+"diffQ.png")
#         #     io.highResPlot2d(diff[2], out_dir+"diffU.png")

        
        
#         teb = enmap.ifft(enmap.map2harm(lensedTQU)).real
#         lT = teb[0]
#         lE = teb[1]
#         lB = teb[2]

        
#         avg_ltt += fmaps.get_simple_power(twrap(lT,T),ones)/Nsims
#         avg_lee += fmaps.get_simple_power(twrap(lE,T),ones)/Nsims
#         avg_lbb += fmaps.get_simple_power(twrap(lB,T),ones)/Nsims
#         avg_lte += fmaps.get_simple_power(twrap(lT,T),ones,twrap(lE,T),ones)/Nsims
#         if k==0:
#             io.highResPlot2d(lE, out_dir+"lensedE.png")
#             io.highResPlot2d(lB, out_dir+"lensedB.png")

#         ft = fmaps.deconvolveBeam(fmaps.convolveBeam(teb[0],modLMap,beamTemplate) + ggenT.getMap(cmbellmax),modLMap,beamTemplate,lowPass=cmbellmax,returnFTOnly = True)
#         fe = fmaps.deconvolveBeam(fmaps.convolveBeam(teb[1],modLMap,beamTemplate) + ggenP.getMap(cmbellmax),modLMap,beamTemplate,lowPass=cmbellmax,returnFTOnly = True)
#         fb = fmaps.deconvolveBeam(fmaps.convolveBeam(teb[2],modLMap,beamTemplate) + ggenP.getMap(cmbellmax),modLMap,beamTemplate,lowPass=cmbellmax,returnFTOnly = True)
    
        
        
        

        

#         print "Reconstructing" , k , " ..."
#         qest.updateTEB_X(ft,fe,-fb,alreadyFTed=True) # !!!
#         qest.updateTEB_Y()

#         for polcomb in polcombs:
#             kappa = enmap.samewcs(qest.getKappa(polcomb).real,thetaMap)
#             if k==0:
#                 Nlkk2d = qest.N.Nlkk[polcomb]+inp_power_2d
#                 bin_edges = np.arange(kellmin,kellmax,100)
#                 ls, npow = binner.bin(Nlkk2d)
#                 npows[polcomb] = npow.copy()
#             auto_power_2d = fmaps.get_simple_power(twrap(kappa,T),ones)
#             ls, auto = binner.bin(auto_power_2d)
#             autos[polcomb].append(auto.copy())

#             kappa_recon_lm = T.copy()
#             kappa_recon_lm.data = kappa
#             cross_power_2d = fmaps.get_simple_power(inp_kappa_lm,ones,kappa_recon_lm,ones)
#             ls,cross_power = binner.bin(cross_power_2d)
#             cross_powers[polcomb].append(cross_power.copy())

#             if k==0: io.highResPlot2d(kappa, out_dir+"recon"+polcomb+".png")
#             kappa_avg[polcomb] += kappa.copy()/Nsims
        
        

#     fkellmin = 300
#     fkellmax = 600
#     kappaMapOrig = thetaMap*0.+fmaps.filter_map(kappaMap,kappaMap.copy()*0.+1.,modLMap,lowPass=fkellmax,highPass=fkellmin)
#     io.highResPlot2d(kappaMapOrig, out_dir+"orig.png")

#     cross_stats = {}
#     auto_stats = {}
#     for polcomb in polcombs:
#         kappa_avg[polcomb] = thetaMap*0.+fmaps.filter_map(kappa_avg[polcomb],kappaMap.copy()*0.+1.,modLMap,lowPass=fkellmax,highPass=fkellmin)
#         io.highResPlot2d(kappa_avg[polcomb], out_dir+"reconavg"+polcomb+".png")

#         cross_stats[polcomb] = getStats(cross_powers[polcomb])
#         auto_stats[polcomb] = getStats(autos[polcomb])

#     ellfine = np.arange(min_ell,kellmax,1)
#     clkk = theory.gCl('kk',ellfine)
#     pl = io.Plotter(scaleY='log')
#     pl.add(ellfine,clkk)
#     for polcomb in polcombs:
#         pl.addErr(ls,cross_stats[polcomb]['mean'],yerr=cross_stats[polcomb]['errmean'],ls="none",marker="o",label=polcomb)
#         pl.add(ls,npows[polcomb],ls="--")
#         pl.add(ls,auto_stats[polcomb]['mean'],ls="none",marker="x")
#     pl.add(ls,input_power,ls="none",marker="x")
#     pl.legendOn()
#     pl.done(out_dir+"cross_power.png")

#     pl = io.Plotter()
#     for polcomb in polcombs:
#         pl.add(ls,(cross_stats[polcomb]['mean']-input_power)*100./input_power,label=polcomb)
#     pl._ax.axhline(y=0.,ls="--")
#     pl.legendOn()
#     pl.done(out_dir+"cross_per.png")

#     ellfine = np.arange(110,8000,1)
#     ls,utt = cmb_binner.bin(avg_utt)
#     ls,ltt = cmb_binner.bin(avg_ltt)
#     pl = io.Plotter(scaleY='log')
#     pl.add(ellfine,ellfine**2.*theory.uCl('TT',ellfine),ls="--")
#     pl.add(ellfine,ellfine**2.*theory.lCl('TT',ellfine))
#     pl.add(ls,utt*ls**2.,ls="none",marker="x",label="u",alpha=0.4)
#     pl.add(ls,ltt*ls**2.,ls="none",marker="x",label="l",alpha=0.4)
#     pl.legendOn(loc="upper right")
#     pl.done(out_dir+"cltt.png")


#     ls,uee = cmb_binner.bin(avg_uee)
#     ls,lee = cmb_binner.bin(avg_lee)
#     pl = io.Plotter(scaleY='log')
#     pl.add(ellfine,ellfine**2.*theory.uCl('EE',ellfine),ls="--")
#     pl.add(ellfine,ellfine**2.*theory.lCl('EE',ellfine))
#     pl.add(ls,uee*ls**2.,ls="none",marker="x",label="u",alpha=0.4)
#     pl.add(ls,lee*ls**2.,ls="none",marker="x",label="l",alpha=0.4)
#     pl.legendOn(loc="upper right")
#     pl.done(out_dir+"clee.png")

#     ls,ubb = cmb_binner.bin(avg_ubb)
#     ls,lbb = cmb_binner.bin(avg_lbb)
#     pl = io.Plotter()
#     pl.add(ellfine,ellfine**2.*theory.uCl('BB',ellfine),ls="--")
#     pl.add(ellfine,ellfine**2.*theory.lCl('BB',ellfine))
#     pl.add(ls,ubb*ls**2.,ls="none",marker="x",label="u",alpha=0.4)
#     pl.add(ls,lbb*ls**2.,ls="none",marker="x",label="l",alpha=0.4)
#     pl.legendOn(loc="upper right")
#     pl.done(out_dir+"clbb.png")


#     ls,ute = cmb_binner.bin(avg_ute)
#     ls,lte = cmb_binner.bin(avg_lte)
#     ute2d = interp1d(ellfine,theory.uCl('TE',ellfine),bounds_error=False,fill_value=0.)(modLMap)
#     lte2d = interp1d(ellfine,theory.lCl('TE',ellfine),bounds_error=False,fill_value=0.)(modLMap)
#     ls,tute = cmb_binner.bin(ute2d)
#     ls,tlte = cmb_binner.bin(lte2d)
    
#     pl = io.Plotter()
#     pl.add(ellfine,ellfine**2.*theory.uCl('TE',ellfine),ls="--")
#     pl.add(ellfine,ellfine**2.*theory.lCl('TE',ellfine))
#     pl.add(ls,tute*ls**2.,ls="none",marker="x",label="u",color="C0")
#     pl.add(ls,tlte*ls**2.,ls="none",marker="x",label="l",color="C1")
#     pl.add(ls,ute*ls**2.,ls="none",marker="o",label="u",alpha=0.3,color="C0")
#     pl.add(ls,lte*ls**2.,ls="none",marker="o",label="l",alpha=0.3,color="C1")
#     pl._ax.set_xlim(2.,3000.)
#     pl.legendOn(loc="upper right")
#     pl.done(out_dir+"clte.png")
    
