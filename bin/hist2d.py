import numpy as np
from peakaboo import liuSims as ls
import orphics.tools.io as io
import os,sys
import orphics.analysis.flatMaps as fmaps

def get_cents(edges):
    return (edges[1:]+edges[:-1])/2.

out_dir = os.environ['WWW']+"peakaboo/"

LC = ls.LiuConvergence(root_dir="/gpfs01/astro/workarea/msyriac/data/jiav2/Maps10/",zstr="1100.00")

cmb = LC.get_kappa(1)


LC = ls.LiuConvergence(root_dir="/gpfs01/astro/workarea/msyriac/data/jiav2/Maps10/",zstr="1.00")

gal = LC.get_kappa(1)


dw = 0.001
bin_edges = np.arange(-0.3,0.3+dw,dw)+dw

hist1d_cmb, bin_edges = np.histogram(cmb.ravel(),bin_edges,density=True)
hist1d_gal, bin_edges = np.histogram(gal.ravel(),bin_edges,density=True)

bincents = get_cents(bin_edges)


pl = io.Plotter()
pl.add(bincents,hist1d_cmb)
pl.add(bincents,hist1d_gal)
pl.done(out_dir+"hist1d.png")


hist2d, x_edges,y_edges = np.histogram2d(cmb.ravel(),gal.ravel(),bin_edges,normed=True)


import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

x = get_cents(x_edges)
y = get_cents(y_edges)
X, Y = np.meshgrid(x, y)
Z = hist2d

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.savefig(out_dir+"hist2d.png")


p2d = fmaps.get_simple_power_enmap(cmb,enmap2=gal)
modlmap = LC.modlmap
import orphics.tools.stats as stats
lbin_edges = np.arange(400,3000,100)
binner = stats.bin2D(modlmap,lbin_edges)
cents, p1d = binner.bin(p2d)

pl = io.Plotter(scaleY='log')
pl.add(cents,p1d)
pl.done(out_dir+"cpower.png")
