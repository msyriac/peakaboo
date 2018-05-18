from __future__ import print_function
import numpy as np
from peakaboo import utils as ls
from orphics import io
import os,sys
import orphics.maps as fmaps
from orphics.mpi import mpi_distribute, MPI
from enlib import enmap, resample
from orphics.stats import bin2D, Stats
import argparse
import cPickle as pickle


# Parse command line
parser = argparse.ArgumentParser(description='Run lensing sims pipe.')
parser.add_argument("InpDir", type=str,help='Input Directory Name (not path, that\'s specified in ini)')
parser.add_argument("OutDir", type=str,help='Output Directory Name')
parser.add_argument("bin_section_power", type=str,help='1d power bin_section')
parser.add_argument("bin_section_hist_1d", type=str,help='1d hist bin_section')
parser.add_argument("bin_sections_hist_2d", type=str,help='2d hist bin_sections cmb,gal')
parser.add_argument("InpDirSmooth", type=str,help='Input directory for map that will be used for deciding smoothing scale')
parser.add_argument("-S", "--seed",     type=int,  default=0,help="Seed for noise.")
parser.add_argument("-N", "--nmax",     type=int,  default=1000,help="Limit to nmax sims.")
parser.add_argument("-G", "--galaxies",     type=str,
                    default=None,help="Comma separated list of galaxy redshifts.")
parser.add_argument("-x", "--smoothings_cmb",     type=str,
                    default=None,help="Comma separated list of CMB smoothing sigma in arcmin.")
parser.add_argument("-y", "--smoothings_gal",     type=str,
                    default=None,help="Comma separated list of gal smoothing sigma in arcmin.")
args = parser.parse_args()



# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

inp_dir = args.InpDir
inp_dir_smooth = args.InpDirSmooth
out_dir = args.OutDir

galzs = [float(x) for x in args.galaxies.split(',')]
if args.smoothings_cmb is None:
    smoothings_cmb = [0.]
else:
    smoothings_cmb = [float(x) for x in args.smoothings_cmb.split(',')]
if args.smoothings_gal is None:
    smoothings_gal = [0.]
else:
    smoothings_gal = [float(x) for x in args.smoothings_gal.split(',')]

PathConfig = io.load_path_config()
io.dout_dir = PathConfig.get("paths","plots")+inp_dir+"/"+out_dir+"/"
result_dir = PathConfig.get("paths","output_data")+inp_dir+"/"+out_dir+"/"
result_dir_smooth = PathConfig.get("paths","output_data")+inp_dir_smooth+"/"+out_dir+"/"
save_dir = PathConfig.get("paths","stats")+inp_dir+"/"+out_dir+"/seed_"+str(args.seed)
io.mkdir(save_dir)

# CMB lens noise
lcents,Nlkk = np.loadtxt(result_dir+"nlkk.txt",unpack=True)
#print(Nlkk)
#print("after loading ",rank, args.InpDir,args.OutDir,len(lcents),len(Nlkk))


file_root = lambda sim_id: result_dir+"kappa_"+str(sim_id).zfill(4)+".fits"
file_root_smooth = lambda sim_id: result_dir_smooth+"kappa_"+str(sim_id).zfill(4)+".fits"

# MPI
Ntot = args.nmax
num_each,each_tasks = mpi_distribute(Ntot,numcores)
mpibox = Stats(comm,tag_start=333)
my_tasks = each_tasks[rank]
if rank==0: print( "At most "+ str(max(num_each)) + " tasks...")

map_root = PathConfig.get("paths","input_data")
LC = ls.LiuConvergence(root_dir=map_root+inp_dir+"/")
LCSmooth = ls.LiuConvergence(root_dir=map_root+inp_dir_smooth+"/")

Config = io.config_from_file("input/recon.ini")
lbin_edges = io.bin_edges_from_config(Config,args.bin_section_power)

hist2d_cmb_bin_section,hist2d_gal_bin_section = args.bin_sections_hist_2d.split(',')

hist_bin_edges_cmb = {}
hist2d_bin_edges_cmb = {}
ihist_bin_edges_cmb = {}
ihist2d_bin_edges_cmb = {}

######### calculate CMB lensing bin edges
for p,scmb in enumerate(smoothings_cmb):
    recon = enmap.read_map(file_root_smooth(0))
    if p==0: shape,wcs = recon.shape, recon.wcs
    if scmb>1.e-5: recon = enmap.smooth_gauss(recon,scmb*np.pi/180./60.)
    sigma_cmb = np.sqrt(np.var(recon))
    hist_bin_edges_cmb[str(scmb)] = io.bin_edges_from_config(Config,args.bin_section_hist_1d)*sigma_cmb
    hist2d_bin_edges_cmb[str(scmb)] = io.bin_edges_from_config(Config,hist2d_cmb_bin_section)*sigma_cmb

    input_k = enmap.ndmap(resample.resample_fft(LC.get_kappa(1,z=1100),shape),wcs)
    isigma_cmb = np.sqrt(np.var(input_k))
    ihist_bin_edges_cmb[str(scmb)] = io.bin_edges_from_config(Config,args.bin_section_hist_1d)*isigma_cmb
    ihist2d_bin_edges_cmb[str(scmb)] = io.bin_edges_from_config(Config,hist2d_cmb_bin_section)*isigma_cmb
    
    
bincents = lambda x: (x[1:]+x[:-1])/2.


ngs = []
hist_bin_edges_gals = []
hist2d_bin_edges_gals = []
ihist_bin_edges_gals = []
ihist2d_bin_edges_gals = []
for z in galzs:
    cov = np.ones((1,1,shape[0],shape[1]))*ls.shear_noise(z)
    ngs.append( fmaps.MapGen(shape,wcs,cov))

######### calculate gal lensing bin edges    
for k,z in enumerate(galzs):
    galkappa = enmap.ndmap(resample.resample_fft(LCSmooth.get_kappa(1,z=z),shape),wcs)
    
    hist_bin_edges_gals.append({})
    hist2d_bin_edges_gals.append({})
    ihist_bin_edges_gals.append({})
    ihist2d_bin_edges_gals.append({})
    for sgal in smoothings_gal:

        galkappa_noisy = galkappa + ngs[k].get_map(seed=(k))
        if sgal>1.e-5: galkappa_noisy = enmap.smooth_gauss(galkappa_noisy,sgal*np.pi/180./60.)
        sigma_gal = np.sqrt(np.var(galkappa_noisy))
        isigma_gal = np.sqrt(np.var(galkappa))
        #print(sigma_gal)
        hist_bin_edges_gals[k][str(sgal)] = io.bin_edges_from_config(Config,args.bin_section_hist_1d)*sigma_gal 
        hist2d_bin_edges_gals[k][str(sgal)] =  io.bin_edges_from_config(Config,hist2d_gal_bin_section )*sigma_gal 
        ihist_bin_edges_gals[k][str(sgal)] = io.bin_edges_from_config(Config,args.bin_section_hist_1d)*isigma_gal 
        ihist2d_bin_edges_gals[k][str(sgal)] =  io.bin_edges_from_config(Config,hist2d_gal_bin_section )*isigma_gal 

####### now do all the statistics
for k,i in enumerate(my_tasks):

    # Read reconstructed CMB lensing
    recon = enmap.read_map(file_root(i))
    if rank==0: print(file_root(i))

    # Get input CMB lens
    input_k = enmap.ndmap(resample.resample_fft(LC.get_kappa(i+1,z=1100),shape),wcs)        

    # Initialize fourier
    if k==0:
        shape,wcs = recon.shape,recon.wcs
        fc = fmaps.FourierCalc(shape,wcs)
        lbinner = bin2D(recon.modlmap(),lbin_edges)
        if i==0: np.save(save_dir+"ALL_power_1d_lbin_edges.npy",lbin_edges)

    # Smooth and calculate 1d CMB pdf
    recon_smoothed = {}
    inputc_smoothed = {}
    for scmb in smoothings_cmb:
        recon_smoothed[str(scmb)] = enmap.smooth_gauss(recon.copy(),scmb*np.pi/180./60.)
        recon_smoothed[str(scmb)] -= recon_smoothed[str(scmb)].mean()
        cmb_pdf,_ = np.histogram(recon_smoothed[str(scmb)].ravel(),hist_bin_edges_cmb[str(scmb)])
        mpibox.add_to_stats("cmb_pdf_%s" % scmb, cmb_pdf)

        inputc_smoothed[str(scmb)] = enmap.smooth_gauss(input_k.copy(),scmb*np.pi/180./60.)
        inputc_smoothed[str(scmb)] -= inputc_smoothed[str(scmb)].mean()
        icmb_pdf,_ = np.histogram(inputc_smoothed[str(scmb)].ravel(),ihist_bin_edges_cmb[str(scmb)])
        mpibox.add_to_stats("icmb_pdf_%s" % scmb, icmb_pdf)
    
        if i==0: np.save(save_dir+"ALL_cmb_pdf_"+str(scmb)+"_bin_edges.npy",hist_bin_edges_cmb[str(scmb)])
        if i==0: np.save(save_dir+"ALL_icmb_pdf_"+str(scmb)+"_bin_edges.npy",ihist_bin_edges_cmb[str(scmb)])
    
    
    # Recon x input
    p2drcic,krc,kic = fc.power2d(recon,input_k)
    cents, prcic = lbinner.bin(p2drcic)

    if k==0:
        # Theory N0
        try:
            assert np.all(np.isclose(cents,lcents))
        except:
            from scipy.interpolate import interp1d
            nlkkfunc = interp1d(lcents,Nlkk,bounds_error=False,fill_value="extrapolate")
            Nlkk = nlkkfunc(cents)

    # Input x Input    
    p2dicic  = fc.f2power(kic,kic)
    cents, picic = lbinner.bin(p2dicic)
    #np.save(save_dir+"icmbXicmb_"+str(i).zfill(4)+".npy",picic)
    mpibox.add_to_stats("icmbXicmb" , picic)
    # Recon x Recon
    p2drcrc  = fc.f2power(krc,krc)
    cents, prcrc = lbinner.bin(p2drcrc)
    #np.save(save_dir+"cmbXcmb_"+str(i).zfill(4)+".npy",prcrc)
    mpibox.add_to_stats("cmbXcmb" , prcrc)

    ###### prep noise maps, so don't need to redo it every loop
    noise_maps = []
    for j,z in enumerate(galzs):
        noise_maps.append( ngs[j].get_map(seed=(i,j,args.seed)) )
        ####### k is realization(my_tasks) conter, i is realization number, 
        ####### j is galaxy redshift counter
        
    # Galaxy lensing
    for j,z in enumerate(galzs):

        # Load noiseless galaxy kappa
        galkappa = enmap.ndmap(resample.resample_fft(LC.get_kappa(i+1,z=z),shape),wcs)
        # Add noise
        galkappa_noisy = galkappa + noise_maps[j]

        
        # input Gal x input CMB lens
        p2dicig,_,_ = fc.power2d(galkappa,input_k)
        cents, picig = lbinner.bin(p2dicig)
        mpibox.add_to_stats("igalXicmb_%.2f" % z , picig)
        
        # noisy Gal x noisy CMB recon
        p2drcig,_,_ = fc.power2d(recon,galkappa_noisy)
        cents, prcig = lbinner.bin(p2drcig)
        mpibox.add_to_stats("galXcmb_%.2f" % z , prcig)


        sgal = smoothings_gal[0]

        # Noisy 1d pdf
        gkappa_noisy = enmap.smooth_gauss(galkappa_noisy.copy(),sgal*np.pi/180./60.) if sgal>1.e-5 else galkappa_noisy.copy()
        gkappa_noisy -= gkappa_noisy.mean()
        gal_pdf,_ = np.histogram(gkappa_noisy.ravel(),hist_bin_edges_gals[j][str(sgal)])
        mpibox.add_to_stats("gal_pdf_%.2f_%s" %(z,sgal), gal_pdf)
        if i==0: np.save(save_dir+"ALL_gal_pdf_"+str(z)+"_"+str(sgal)+"_bin_edges.npy",hist_bin_edges_gals[j][str(sgal)])

        # Noiseless 1d pdf
        gkappa = enmap.smooth_gauss(galkappa.copy(),sgal*np.pi/180./60.) if sgal>1.e-5 else galkappa.copy()
        gkappa -= gkappa.mean()
        gal_pdf,_ = np.histogram(gkappa.ravel(),ihist_bin_edges_gals[j][str(sgal)])
        #np.save(save_dir+"igal_pdf_"+str(z)+"_"+str(sgal)+"_"+str(i).zfill(4)+".npy",gal_pdf)
        mpibox.add_to_stats("igal_pdf_%.2f_%s" %(z,sgal), gal_pdf)
        if i==0: np.save(save_dir+"ALL_igal_pdf_"+str(z)+"_"+str(sgal)+"_bin_edges.npy",ihist_bin_edges_gals[j][str(sgal)])

        # Noisy 2D CMB pdf
        scmb = smoothings_cmb[0]
        pdf_2d,_,_ = np.histogram2d(recon_smoothed[str(scmb)].ravel(),gkappa_noisy.ravel(),bins=(hist2d_bin_edges_cmb[str(scmb)],hist2d_bin_edges_gals[j][str(sgal)]))
        #np.save(save_dir+"galXcmb_2dpdf_"+str(z)+"_"+str(sgal)+"_"+str(scmb)+"_"+str(i).zfill(4)+".npy",pdf_2d)
        mpibox.add_to_stats("galXcmb_2dpdf_%.2f_%s_%s" %(z,sgal,scmb), pdf_2d)
        if i==0: pickle.dump((hist2d_bin_edges_cmb[str(scmb)],hist2d_bin_edges_gals[j][str(sgal)]),open(save_dir+"ALL_galXcmb_pdf_"+str(z)+"_"+str(sgal)+"_"+str(scmb)+"_bin_edges.pkl",'wb'))

        # Noiseless 2D CMB pdf
        scmb = smoothings_cmb[0]
        pdf_2d,_,_ = np.histogram2d(inputc_smoothed[str(scmb)].ravel(),gkappa.ravel(),bins=(ihist2d_bin_edges_cmb[str(scmb)],ihist2d_bin_edges_gals[j][str(sgal)]))
        #np.save(save_dir+"igalXicmb_2dpdf_"+str(z)+"_"+str(sgal)+"_"+str(scmb)+"_"+str(i).zfill(4)+".npy",pdf_2d)
        mpibox.add_to_stats("igalXicmb_2dpdf_%.2f_%s_%s" %(z,sgal,scmb), pdf_2d)
        if i==0: pickle.dump((ihist2d_bin_edges_cmb[str(scmb)],ihist2d_bin_edges_gals[j][str(sgal)]),open(save_dir+"ALL_igalXicmb_pdf_"+str(z)+"_"+str(sgal)+"_"+str(scmb)+"_bin_edges.pkl",'wb'))
        
        for z2 in galzs[j:]:
            m = galzs.index(z2)
            # Load noiseless galaxy kappa
            galkappa2 = enmap.ndmap(resample.resample_fft(LC.get_kappa(i+1,z=z2),shape),wcs)
            # Add noise
            galkappa_noisy2 = galkappa2 + noise_maps[m]


            # input Gal x Gal
            p2digig,_,_ = fc.power2d(galkappa,galkappa2)
            cents, prigig = lbinner.bin(p2digig)
            mpibox.add_to_stats("igalXigal_%.2f_%.2f" %(z,z2), prigig)

            
            # noisy Gal x noisy Gal
            p2digig,_,_ = fc.power2d(galkappa_noisy,galkappa_noisy2)
            cents, prigig = lbinner.bin(p2digig)
            mpibox.add_to_stats("galXgal_%.2f_%.2f" %(z,z2), prigig)


            # Smooth
            gkappa_noisy2 = enmap.smooth_gauss(galkappa_noisy2.copy(),sgal*np.pi/180./60.) if sgal>1.e-5 else galkappa_noisy2.copy()
            gkappa_noisy2 -= gkappa_noisy2.mean()
            gkappa2 = enmap.smooth_gauss(galkappa2.copy(),sgal*np.pi/180./60.) if sgal>1.e-5 else galkappa2.copy()
            gkappa2 -= gkappa2.mean()
            
            # Noisy 2D cross gal pdf
            pdf_2d,_,_ = np.histogram2d(gkappa_noisy.ravel(),gkappa_noisy2.ravel(),bins=(hist2d_bin_edges_gals[j][str(sgal)],hist2d_bin_edges_gals[m][str(sgal)]))
            mpibox.add_to_stats("galXgal_2dpdf_%.2f_%.2f_%s" %(z,z2,sgal), pdf_2d)
            if i==0: pickle.dump((hist2d_bin_edges_gals[j][str(sgal)],hist2d_bin_edges_gals[m][str(sgal)]),open(save_dir+"ALL_galXgal_pdf_"+str(z)+"_"+str(z2)+"_"+str(sgal)+"_bin_edges.pkl",'wb'))

            # Noiseless 2D cross gal pdf
            pdf_2d,_,_ = np.histogram2d(gkappa.ravel(),gkappa2.ravel(),bins=(ihist2d_bin_edges_gals[j][str(sgal)],ihist2d_bin_edges_gals[m][str(sgal)]))
            mpibox.add_to_stats("igalXigal_2dpdf_%.2f_%.2f_%s" %(z,z2,sgal), pdf_2d)
            if i==0: pickle.dump((ihist2d_bin_edges_gals[j][str(sgal)],ihist2d_bin_edges_gals[m][str(sgal)]),open(save_dir+"ALL_igalXigal_pdf_"+str(z)+"_"+str(z2)+"_"+str(sgal)+"_bin_edges.pkl",'wb'))

    ####### what are these for? plotting?
    mpibox.add_to_stats("icig",picig) # input cmb iput gal
    mpibox.add_to_stats("rcig",prcig) # recon cmb input gal
    mpibox.add_to_stats("icic",picic) # input input cmb
    mpibox.add_to_stats("rcic",prcic) # recon input cmb
    mpibox.add_to_stats("rcrc",prcrc) # recon recon cmb

    if rank==0 and (k+1)%1==0: print( "Rank 0 done with "+str(k+1)+ " / "+str( len(my_tasks))+ " tasks.")
    


mpibox.get_stats(verbose=False)

if rank==0:
    print ('collecting results')
    
    ############ start collecting stats
    scmb = smoothings_cmb[0]
    arr = mpibox.vectors["cmb_pdf_%s" % scmb]
    np.save(save_dir+"ALL_cmb_pdf_"+str(scmb)+".npy",arr)
    
    del arr
    arr = mpibox.vectors["icmb_pdf_%s" % scmb]
    np.save(save_dir+"ALL_icmb_pdf_"+str(scmb)+".npy",arr)
        
    del arr
    arr = mpibox.vectors["icmbXicmb"]
    np.save(save_dir+"ALL_icmbXicmb.npy",arr)
    
    del arr
    arr = mpibox.vectors["cmbXcmb"]
    np.save(save_dir+"ALL_cmbXcmb.npy",arr)

    
    for j,z in enumerate(galzs):
        #### cross power spectrum
        del arr
        arr = mpibox.vectors["igalXicmb_%.2f" % z]
        np.save(save_dir+"ALL_igalXicmb_"+str(z)+".npy",arr)
        del arr
        arr = mpibox.vectors["galXcmb_%.2f" % z]
        np.save(save_dir+"ALL_galXcmb_"+str(z)+".npy",arr)

        sgal = smoothings_gal[0]
        scmb = smoothings_cmb[0]

        # 1d pdf
        del arr
        arr = mpibox.vectors["gal_pdf_%.2f_%s" %(z,sgal)] 
        np.save(save_dir+"ALL_gal_pdf_z"+str(z)+"_s"+str(sgal)+".npy",arr)
        del arr
        arr = mpibox.vectors["igal_pdf_%.2f_%s" %(z,sgal)]
        np.save(save_dir+"ALL_igal_pdf_z"+str(z)+"_s"+str(sgal)+".npy",arr)

        # 2d cmb pdf
        del arr
        arr = mpibox.vectors["galXcmb_2dpdf_%.2f_%s_%s" %(z,sgal,scmb)]
        np.save(save_dir+"ALL_galXcmb_2dpdf_z"+str(z)+"_sg"+str(sgal)+"_sc"+str(scmb)+".npy",arr)
        del arr
        arr = mpibox.vectors["igalXicmb_2dpdf_%.2f_%s_%s" %(z,sgal,scmb)]
        np.save(save_dir+"ALL_igalXicmb_2dpdf_z"+str(z)+"_sg"+str(sgal)+"_sc"+str(scmb)+".npy",arr)
        
        
        for m,z2 in enumerate(galzs[j:]):
            del arr
            arr = mpibox.vectors["igalXigal_%.2f_%.2f" %(z,z2)]
            np.save(save_dir+"ALL_igalXigal_"+str(z)+"_"+str(z2)+".npy",arr)
            
            del arr
            arr = mpibox.vectors["galXgal_%.2f_%.2f" %(z,z2)]
            np.save(save_dir+"ALL_galXgal_"+str(z)+"_"+str(z2)+".npy",arr)

            del arr
            arr = mpibox.vectors["galXgal_2dpdf_%.2f_%.2f_%s" %(z,z2,sgal)]
            np.save(save_dir+"ALL_galXgal_2dpdf_z"+str(z)+"_z2"+str(z2)+"_sg"+str(sgal)+".npy",arr)

            del arr
            arr = mpibox.vectors["igalXigal_2dpdf_%.2f_%.2f_%s" %(z,z2,sgal)]
            np.save(save_dir+"ALL_igalXigal_2dpdf_z"+str(z)+"_z2"+str(z2)+"_sg"+str(sgal)+".npy",arr)

        


    ########## make sanity check plots
    rcrc = mpibox.stats["rcrc"]["mean"]
    icig = mpibox.stats["icig"]["mean"]
    rcig = mpibox.stats["rcig"]["mean"]
    rcig_err = mpibox.stats["rcig"]["err"]
    inputk = mpibox.stats["icic"]["mean"]
    cross = mpibox.stats["rcic"]["mean"]
    cross_err = mpibox.stats["rcic"]["err"]

    os.system('mkdir -pv %s'%(io.dout_dir))
    pdiff = (cross-inputk)/inputk
    
    pl = io.Plotter(xlabel="$\\ell$",ylabel="$\\Delta C_{\ell}/C_{\ell}$")
    pl.add(cents,pdiff,color="k")
    pl.hline()
    #pl._ax.set_ylim(-2,1)
    pl._ax.set_xlim(lbin_edges[0],lbin_edges[-1])
    pl.done(io.dout_dir+"pdiff.png")

    pl = io.Plotter(xlabel="$\\ell$",ylabel="$C_{\ell}$",yscale='log')
    pl.add(cents,inputk,color="k",label="ixi")
    pl.add(cents,Nlkk,ls="--",label="nlkk")
    pl.add(cents,Nlkk+inputk,ls="-",label="nlkk+clkk")
    pl.add(cents,rcrc,marker="o",ls="none",label="rxr")
    pl.add_err(cents,cross,yerr=cross_err,marker="o",label="rxi")
    pl._ax.set_xlim(lbin_edges[0],lbin_edges[-1])
    pl.legend()
    pl.done(io.dout_dir+"clkk.png")

    
    pl = io.Plotter(xlabel="$\\ell$",ylabel="$\\ell C_{\ell}$")
    pl.add(cents,icig*cents,color="k")
    pl.add_err(cents,rcig*cents,yerr=rcig_err*cents,marker="o")
    pl._ax.set_xlim(lbin_edges[0],lbin_edges[-1])
    pl.hline()
    pl.done(io.dout_dir+"galcross.png")
    
    # Delete everything else
    
    # cmd = "rm %s*_????.npy"%(save_dir)
    # #print (cmd)    
    # os.system(cmd)
