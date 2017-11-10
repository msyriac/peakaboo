import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
from alhazen.quadraticEstimator import Estimator
import alhazen.io as aio
import alhazen.lensTools as lt
from enlib import enmap, resample, lensing, fft
from orphics.tools.mpi import mpi_distribute, MPIStats
import numpy as np
import logging, time, os
import astropy.io.fits as fits

class LiuConvergence(object):

    def __init__(self,root_dir="/gpfs01/astro/workarea/msyriac/data/jiav2/massless/",zstr="1100.00"):


        self.root = root_dir
        self.kappa_file = lambda i :root_dir+"WLconv_z"+zstr+"_"+str(i).zfill(4)+"r.fits"

        
        size_deg = 3.5
        Npix = 2048.
            
        px = size_deg*60./Npix
        self.px = px

        self.shape, self.wcs = enmap.rect_geometry(size_deg*60.,px,proj="CAR",pol=False)

        self.zstr = zstr
        self.lxmap,self.lymap,self.modlmap,self.angmap,self.lx,self.ly = fmaps.get_ft_attributes_enmap(self.shape,self.wcs)

        
    def get_kappa(self,index):

        
        my_map = fits.open(self.kappa_file(index))[0]
        my_map = my_map.data
        
        try:
            assert my_map.shape == self.shape
        except:
            print my_map.shape
            print self.shape
            print "ERROR"
            sys.exit()

        self.low_pass_ell = 10000
        retmap = enmap.ndmap(my_map,self.wcs)
        retmap = enmap.ndmap(fmaps.filter_map(retmap,retmap.copy()*0.+1.,self.modlmap,lowPass=self.low_pass_ell),self.wcs)

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
                 bin_edges=None,verbose=False):
        
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
            self.logger = io.get_logger("peakaboo_recon")


        sims_section = "sims_liu"
        analysis_section = "analysis_liu"

        if verbose and self.rank==0: self.logger.info("Initializing patches...")
        shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(self.Config,
                                                                        sims_section,
                                                                        analysis_section,
                                                                        pol=pol)

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
            search_path = map_root+inp_dir+"/WLconv_z1100*.fits"
            files = glob.glob(search_path)
            Ntot = len(files)
            assert Ntot>0
            
        self.lc =  LiuConvergence(root_dir=map_root+inp_dir+"/",zstr="1100.00")

            
        num_each,each_tasks = mpi_distribute(Ntot,self.numcores)
        self.mpibox = MPIStats(self.comm,num_each,tag_start=333)
        if self.rank==0: self.logger.info( "At most "+ str(max(num_each)) + " tasks...")
        self.sim_ids = each_tasks[self.rank]

        if verbose and self.rank==0: self.logger.info( "Initializing cosmology...")


        cosmology_section = "cc_default" #!!!!!!!!WRONG
        
        def do_cosmology():
            return aio.theory_from_config(self.Config,cosmology_section,dimensionless=False)

        if self.rank==0:
            try:
                old_cores = os.environ["OMP_NUM_THREADS"]
            except:
                old_cores = "1"
            import multiprocessing
            num_cores= str(multiprocessing.cpu_count())
            os.environ["OMP_NUM_THREADS"] = num_cores
            self.logger.info( "Rank 0 possibly calling CAMB with "+num_cores+" cores...")
            theory, cc, lmax = do_cosmology()
            os.environ["OMP_NUM_THREADS"] = old_cores
            self.logger.info( "Rank 0 done with CAMB and setting OMP_NUM_THREADS back to  "+old_cores)

        self.comm.Barrier()
        if self.rank!=0:
            theory, cc, lmax = do_cosmology()
        
        self.pdat.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)
        self.psim.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)

        self.lens_order = self.Config.getint(sims_section,"lens_order")
        self.map_root = self.lc.root



        # RECONSTRUCTION INIT
        if verbose and self.rank==0: self.logger.info( "Initializing quadratic estimator...")

        min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)
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
        lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
        lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)

        template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
        fMaskCMB_TX = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=tellminX,lmax=tellmaxX)
        fMaskCMB_TY = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=tellminY,lmax=tellmaxY)
        fMaskCMB_PX = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=pellminX,lmax=pellmaxX)
        fMaskCMB_PY = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=pellminY,lmax=pellmaxY)
        fMask = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=kellmin,lmax=kellmax)

        with io.nostdout():
            self.qestimator = Estimator(template_dat,
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


        if verbose and self.rank==0: self.logger.info( "Initializing binner...")
        import orphics.tools.stats as stats
        
        if bin_edges is None:
            bin_edges = np.arange(200,4000,100)

        self.lbinner = stats.bin2D(self.pdat.modlmap,bin_edges)
        self.fc = enmap.FourierCalc(self.pdat.shape,self.pdat.wcs)
        self.cents = self.lbinner.centers
        self.pwrfunc = lambda x: self.lbinner.bin(self.fc.power2d(x)[0])[1]
        
        self.plot_dir = PathConfig.get("paths","plots")+inp_dir+"/"+out_dir+"/"
        self.result_dir = PathConfig.get("paths","output_data")+inp_dir+"/"+out_dir+"/"
        io.mkdir(self.result_dir)


    def get_unlensed(self,seed):
        return self.psim.get_unlensed_cmb(seed=seed)

    def get_kappa(self,index,stack=False):
        imap = self.lc.get_kappa(index+1)

        retmap = enmap.ndmap(resample.resample_fft(imap,self.psim.shape[-2:]),self.psim.wcs)if imap.shape!=self.psim.shape[-2:] \
                            else enmap.ndmap(imap,self.psim.wcs)
        return enmap.ndmap(fmaps.filter_map(retmap,retmap*0.+1.,self.psim.modlmap,lowPass=self.kellmax,highPass=self.kellmin),self.psim.wcs)
        

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
        self.qestimator.updateTEB_X(lT,lE,lB,alreadyFTed=True)
        self.qestimator.updateTEB_Y()
        recon = self.qestimator.getKappa(self.estimator).real
        return recon


    def save_kappa(self,kappa,sim_id):
        kappa = enmap.ndmap(kappa,self.pdat.wcs)
        enmap.write_map(self.result_dir+"kappa_"+str(sim_id).zfill(4)+".fits",kappa)
    
    def dump(self):
        # noiseless = self.mpibox.stacks["noiseless"]
        # noisy = self.mpibox.stacks["noise"]
        # ells = self.cents
        # pl = io.Plotter(scaleY='log')
        # pl.add(ells,noiseless*ells**2.,label="noiseless")
        # pl.add(ells,noisy*ells**2.,label="noisy")
        # pl.legendOn()
        # pl.done(self.plot_dir+"cls.png")
        self.logger.info( "Done!")
        
    def save_cache(self,lensed,sim_id):
        np.save(self.map_root+"lensed_cmb_"+str(sim_id)+".npy",lensed)
    def load_cached(self,sim_id):
        return np.load(self.map_root+"lensed_cmb_"+str(sim_id)+".npy")
