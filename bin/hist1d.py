import numpy as np
from peakaboo import liuSims as ls
import orphics.tools.io as io
import os,sys
import orphics.analysis.flatMaps as fmaps

def get_cents(edges):
    return (edges[1:]+edges[:-1])/2.

out_dir = os.environ['WWW']+"peakaboo/"

LC = ls.LiuConvergence(root_dir="/gpfs01/astro/workarea/msyriac/data/jiav2/massless/",zstr="1100.00")

cmb1 = LC.get_kappa(1)


LC = ls.LiuConvergence(root_dir="/gpfs01/astro/workarea/msyriac/data/jiav2/massive/",zstr="1100.00")

cmb2 = LC.get_kappa(1)


dw = 0.01
bin_edges = np.arange(-0.3,0.3+dw,dw)+dw

hist1d_cmb1, bin_edges = np.histogram(cmb1.ravel(),bin_edges,density=False)
hist1d_cmb2, bin_edges = np.histogram(cmb2.ravel(),bin_edges,density=False)

bincents = get_cents(bin_edges)


pl = io.Plotter()
pl.add(bincents,hist1d_cmb1,label="massless")
pl.add(bincents,hist1d_cmb2,label="massive")
pl.legendOn()
pl.done(out_dir+"hist1dcmb.png")

pl = io.Plotter()
pl.add(bincents,(hist1d_cmb2-hist1d_cmb1)*100./hist1d_cmb1)
pl._ax.set_ylim(-20.,20.)
pl.done(out_dir+"hist1dcmb_pdiff.png")


