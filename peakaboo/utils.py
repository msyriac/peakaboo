import orphics.io as io
import orphics.maps as maps
from orphics.lensing import Estimator
import alhazen.io as aio
import alhazen.lensTools as lt
from enlib import enmap, resample, lensing, fft
from orphics.mpi import mpi_distribute
from orphics.stats import Stats as MPIStats
import numpy as np
import logging, time, os
import astropy.io.fits as fits
from orphics import cosmology


# DEFINE SHEAR NOISE MODEL HERE
def shear_noise(z, shape_noise = 0.3):
    z_arr = np.arange(0.5,3,0.5)    
    ngal_arr = np.array([8.83, 13.25, 11.15, 7.36, 4.26])
    if z in z_arr:
        ngal = ngal_arr[np.where(z_arr==z)]
    else:
        ngal = 0
#ngal=None
#    ngal=8.83 if z==0.5 else None
#    ngal=13.25 if z==1.0 else None
#    ngal=11.15 if z==1.5 else None
#    ngal=7.36 if z==2.0 else None
#    ngal=4.26 if z==2.5 else None
#    assert ngal is not None
    return shape_noise**2./(ngal*1.18e7)


class LiuConvergence(object):

    def __init__(self,root_dir="/gpfs01/astro/workarea/msyriac/data/sims/jia/"):
        self.root = root_dir
        size_deg = 3.5
        Npix = 2048.
        px = size_deg*60./Npix
        self.px = px
        self.shape, self.wcs = maps.rect_geometry(width_deg = size_deg,px_res_arcmin=px,proj="CAR",pol=False)
        self.modlmap = enmap.modlmap(self.shape,self.wcs)
        #print ('JIA print self.shape',self.shape)
        
    def get_kappa(self,index,z=1100):
        zstr = "{:.2f}".format(z)
        kappa_file = self.root+"Maps%02d"%(z*10)+"/WLconv_z"+zstr+"_"+str(index).zfill(4)+"r.fits"
        
        my_map = fits.open(kappa_file)[0]
        my_map = my_map.data
        #print ('JIA print my_map.shape',my_map.shape)

        assert my_map.shape == self.shape
        low_pass_ell = 10000
        retmap = enmap.ndmap(my_map,self.wcs)
        kmask = maps.mask_kspace(self.shape,self.wcs,lmax=low_pass_ell)
        retmap = enmap.ndmap(maps.filter_map(retmap,kmask),self.wcs)
        #print ('JIA print retmap.shape', retmap.shape)
        return retmap


def theory_from_camb_file(camb_file):

    from orphics.theory.cambCall import cambInterface
    cint = cambInterface(outName,templateIni,cambRoot,option=0,seed=0)

    cint.setParam('get_scalar_cls' , 'T')
    cint.setParam('get_transfer' , 'F')
    cint.setParam('do_nonlinear' , '3')

    cint.call()



class PeakabooPipeline(object):

    def __init__(self,estimator,PathConfig,inp_dir,out_dir,Nmax,recon_section,
                 experiment,recon_config_file="input/recon.ini",
                 mpi_comm=None,
                 bin_section=None,verbose=False,debug=False):
        
        self.Config = io.config_from_file(recon_config_file)
        assert estimator=="TT" or estimator=="EB"
        pol = True if estimator=="EB" else False
        self.estimator = estimator

        # Get MPI comm
        self.comm = mpi_comm
        try:
            self.rank = mpi_comm.Get_rank()
            self.numcores = mpi_comm.Get_size()
        except:
            self.rank = 0
            self.numcores = 1

        if self.rank==0:
            #print 'JIA: print I am rank 0' 
            #self.logger = io.get_logger("recon")
            class hack:                                                                                                                              
               def info(self,x):                                                                                                                    
                   print(x)                                                                                                                        
                                                                                                                                                    
            T = hack()                                                                                                                              
            self.logger = T

        sims_section = "sims_liu"
        analysis_section = "analysis_liu"

        if verbose and self.rank==0: self.logger.info("Initializing patches...")
        shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(self.Config,
                                                                        sims_section,
                                                                        analysis_section,
                                                                        pol=pol)
        #print ('JIA print shape_sim, shape_dat', shape_sim, shape_dat)
        if verbose and self.rank==0: self.logger.info("Got shapes and wcs.")
        
        self.psim = aio.patch_array_from_config(self.Config,experiment,
                                                shape_sim,wcs_sim,dimensionless=False,skip_instrument=True)
        self.pdat = aio.patch_array_from_config(self.Config,experiment,
                                                shape_dat,wcs_dat,dimensionless=False)

        if verbose and self.rank==0: self.logger.info("Initialized PatchArrays.")


        if verbose and self.rank==0: self.logger.info("Getting num tasks...")

        
        map_root = PathConfig.get("paths","input_data")
        if Nmax is not None:
            Ntot = Nmax
        else:
            import glob
            search_path = map_root+inp_dir+"/Maps11000/WLconv_z1100*.fits"
            files = glob.glob(search_path)
            #print files
            #print 'JIA NOTE: SEARCH PATH',search_path
            Ntot = len(files)
            #print Ntot
            assert Ntot>0

        #print 'before self.lc'
        self.lc =  LiuConvergence(root_dir=map_root+inp_dir+"/")
        #print 'after self.lc'
            
        num_each,each_tasks = mpi_distribute(Ntot,self.numcores)
        self.mpibox = MPIStats(self.comm,tag_start=333)
        if self.rank==0: self.logger.info( "At most "+ str(max(num_each)) + " tasks...")
        self.sim_ids = each_tasks[self.rank]

        if verbose and self.rank==0: self.logger.info( "Initializing cosmology...")


        # cosmology_section = "cc_default" #!!!!!!!!WRONG
        
        # def do_cosmology():
        #     return aio.theory_from_config(self.Config,cosmology_section,dimensionless=False)

        # if self.rank==0:
        #     try:
        #         old_cores = os.environ["OMP_NUM_THREADS"]
        #     except:
        #         old_cores = "1"
        #     import multiprocessing
        #     num_cores= str(multiprocessing.cpu_count())
        #     os.environ["OMP_NUM_THREADS"] = num_cores
        #     self.logger.info( "Rank 0 possibly calling CAMB with "+num_cores+" cores...")
        #     theory, cc, lmax = do_cosmology()
        #     os.environ["OMP_NUM_THREADS"] = old_cores
        #     self.logger.info( "Rank 0 done with CAMB and setting OMP_NUM_THREADS back to  "+old_cores)

        # self.comm.Barrier()
        # if self.rank!=0:
        #     theory, cc, lmax = do_cosmology()

        cosmo_name = inp_dir.split('/')[0]
        camb_names,cosmologies = np.loadtxt("input/camb/fn_mapping.txt",dtype=str,unpack=True)
        this_camb = camb_names[cosmologies==cosmo_name]
        assert this_camb.size==1
        this_camb = "camb_"+this_camb.ravel()[0]
        #this_camb = "camb_mnv0.00000_om0.30000_As2.1000"
        cc = None
        lmax = 6000
        theory = cosmology.loadTheorySpectraFromCAMB('input/camb/'+this_camb,
                                             unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

        
        self.pdat.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)
        self.psim.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)

        self.lens_order = self.Config.getint(sims_section,"lens_order")



        # RECONSTRUCTION INIT
        if verbose and self.rank==0: self.logger.info( "Initializing quadratic estimator...")

        min_ell = maps.minimum_ell(shape_dat,wcs_dat)
        lb = aio.ellbounds_from_config(self.Config,recon_section,min_ell)
        tellminY = lb['tellminY']
        tellmaxY = lb['tellmaxY']
        pellminY = lb['pellminY']
        pellmaxY = lb['pellmaxY']
        tellminX = lb['tellminX']
        tellmaxX = lb['tellmaxX']
        pellminX = lb['pellminX']
        pellmaxX = lb['pellmaxX']
        kellmin = lb['kellmin']
        kellmax = lb['kellmax']
        self.kellmin = kellmin
        self.kellmax = kellmax
        lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = maps.get_ft_attributes(shape_dat,wcs_dat)
        lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = maps.get_ft_attributes(shape_sim,wcs_sim)

        fMaskCMB_TX = maps.mask_kspace(shape_dat,wcs_dat,lmin=tellminX,lmax=tellmaxX)
        fMaskCMB_TY = maps.mask_kspace(shape_dat,wcs_dat,lmin=tellminY,lmax=tellmaxY)
        fMaskCMB_PX = maps.mask_kspace(shape_dat,wcs_dat,lmin=pellminX,lmax=pellmaxX)
        fMaskCMB_PY = maps.mask_kspace(shape_dat,wcs_dat,lmin=pellminY,lmax=pellmaxY)
        fMask = maps.mask_kspace(shape_dat,wcs_dat,lmin=kellmin,lmax=kellmax)
        self.fMaskSim = maps.mask_kspace(shape_sim,wcs_sim,lmin=kellmin,lmax=kellmax)

        with io.nostdout():
            self.qestimator = Estimator(shape_dat,wcs_dat,
                                        theory,
                                        theorySpectraForNorm=None,
                                        noiseX2dTEB=[self.pdat.nT,self.pdat.nP,self.pdat.nP],
                                        noiseY2dTEB=[self.pdat.nT,self.pdat.nP,self.pdat.nP],
                                        fmaskX2dTEB=[fMaskCMB_TX,fMaskCMB_PX,fMaskCMB_PX],
                                        fmaskY2dTEB=[fMaskCMB_TY,fMaskCMB_PY,fMaskCMB_PY],
                                        fmaskKappa=fMask,
                                        kBeamX = self.pdat.lbeam,
                                        kBeamY = self.pdat.lbeam,
                                        doCurl=False,
                                        TOnly=not(pol),
                                        halo=True,
                                        uEqualsL=True,
                                        gradCut=None,verbose=False,
                                        bigell=lmax)
            Nlkk2d = self.qestimator.N.Nlkk[self.estimator]

        if verbose and self.rank==0: self.logger.info( "Initializing binner...")
        import orphics.tools.stats as stats

        
        bin_edges = io.bin_edges_from_config(self.Config,bin_section)

        self.lbinner = stats.bin2D(self.pdat.modlmap,bin_edges)
        self.fc = maps.FourierCalc(self.pdat.shape,self.pdat.wcs)
        self.cents = self.lbinner.centers
        
        self.plot_dir = PathConfig.get("paths","plots")+inp_dir+"/"+out_dir+"/"
        self.result_dir = PathConfig.get("paths","output_data")+inp_dir+"/"+out_dir+"/"
        try:
            io.mkdir(self.result_dir)
        except:
            pass

        if debug:
            try:
                io.mkdir(self.plot_dir)
            except:
                pass


        cents,Nlkk = self.lbinner.bin(Nlkk2d)
        io.save_cols(self.result_dir+"nlkk.txt",(cents,Nlkk))

    def get_unlensed(self,seed):
        return self.psim.get_unlensed_cmb(seed=seed)

    def get_kappa(self,index,stack=False):
        imap = self.lc.get_kappa(index+1,z=1100)
       # print ('JIA print self.psim.shape', psim.shape)    

        retmap = enmap.ndmap(resample.resample_fft(imap,self.psim.shape[-2:]),self.psim.wcs)if imap.shape!=self.psim.shape[-2:] \
                            else enmap.ndmap(imap,self.psim.wcs)
        return enmap.ndmap(maps.filter_map(retmap,self.fMaskSim),self.psim.wcs)
        

    def get_lensed(self,unlensed,kappa):
        phi = lt.kappa_to_phi(kappa,self.psim.modlmap,return_fphi=False)
        grad_phi = enmap.grad(phi)
        lensed = lensing.lens_map(unlensed, grad_phi, order=self.lens_order, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7)
        return lensed

    def downsample(self,imap):
        return enmap.ndmap(resample.resample_fft(imap,self.pdat.shape),self.pdat.wcs)  if imap.shape!=self.pdat.shape \
                  else enmap.ndmap(imap,self.pdat.wcs)

    def beam(self,imap):
        flensed = fft.fft(imap,axes=[-2,-1])
        lbeam = self.pdat.lbeam
        flensed *= lbeam
        return fft.ifft(flensed,axes=[-2,-1],normalize=True).real

    def get_noise(self,seed):
        pdat = self.pdat
        return pdat.get_noise_sim(seed=seed)

    def qest(self,lT,lE=None,lB=None):
        #print ('JIA quest lT.shape', lT.shape)
        self.qestimator.updateTEB_X(lT,lE,lB,alreadyFTed=True)
        self.qestimator.updateTEB_Y()
        recon = self.qestimator.get_kappa(self.estimator).real
        #print ('JIA quest recon.shape', recon.shape)
        return recon


    def save_kappa(self,kappa,sim_id):
        kappa = enmap.ndmap(kappa,self.pdat.wcs)
        enmap.write_map(self.result_dir+"kappa_"+str(sim_id).zfill(4)+".fits",kappa)
        #print (self.result_dir+"kappa_"+str(sim_id).zfill(4)+".fits")

    def power_plotter(self,lteb,label):

        if self.estimator=="TT":

            lT = lteb
            tt = self.lbinner.bin(self.fc.f2power(lT,lT))[1]
            self.mpibox.add_to_stack(label+"TT",tt)

        else:
            
            lT = lteb[0]
            lE = lteb[1]
            lB = lteb[2]


            tt = self.lbinner.bin(self.fc.f2power(lT,lT))[1]
            ee = self.lbinner.bin(self.fc.f2power(lE,lE))[1]
            bb = self.lbinner.bin(self.fc.f2power(lB,lB))[1]
            te = self.lbinner.bin(self.fc.f2power(lT,lE))[1]
            tb = self.lbinner.bin(self.fc.f2power(lT,lB))[1]
            eb = self.lbinner.bin(self.fc.f2power(lE,lB))[1]

            self.mpibox.add_to_stack(label+"TT",tt)
            self.mpibox.add_to_stack(label+"EE",ee)
            self.mpibox.add_to_stack(label+"BB",bb)
            self.mpibox.add_to_stack(label+"TE",te)
            self.mpibox.add_to_stack(label+"TB",tb)
            self.mpibox.add_to_stack(label+"EB",eb)

                        
    def dump(self,debug=False):


        if debug:
            ells = self.cents
            # TT, EE, BB

            pl = io.Plotter(yscale='log')
            for spec in ['TT','EE','BB']:
                noiseless = self.mpibox.stacks["lensed"+spec]
                noisy = self.mpibox.stacks["observed-deconvolved"+spec]
            
                pl.add(ells,noiseless*ells**2.)
                pl.add(ells,noisy*ells**2.,ls="--")
            pl.legend()
            pl.done(self.plot_dir+"clsauto.png")


            # TE

            pl = io.Plotter()
            for spec in ['TE']:
                noiseless = self.mpibox.stacks["lensed"+spec]
                noisy = self.mpibox.stacks["observed-deconvolved"+spec]
            
                pl.add(ells,noiseless*ells**2.)
                pl.add(ells,noisy*ells**2.,ls="--")
            pl.legend()
            pl.done(self.plot_dir+"clste.png")



            # TB, EB


            pl = io.Plotter()
            for spec in ['EB','TB']:
                noiseless = self.mpibox.stacks["lensed"+spec]
                noisy = self.mpibox.stacks["observed-deconvolved"+spec]
            
                pl.add(ells,noiseless*ells**2.)
                pl.add(ells,noisy*ells**2.,ls="--")
            pl.legend()
            pl.done(self.plot_dir+"clsnull.png")

        
        self.logger.info( "Done!")
       
 
