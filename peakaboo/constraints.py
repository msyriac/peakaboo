from scipy import *
import numpy as np
import WLanalysis
from emcee.utils import MPIPool 
import sys, itertools
import emcee

Nk='10k' # '5ka', '5kb'
Nmin=1e-3
testfn = 'collapsed2'
Ngrid = 50
Nchain = 1000
try:
    Nk = str(sys.argv[1])
except Exception:
    pass

z_arr = arange(0.5,3,0.5)
Nz = len(z_arr)
#####################################
######## set up folders #############
#####################################

######## stampede2
stats_dir = '/scratch/02977/jialiu/peakaboo/'
ebcov_dir = stats_dir+'stats/Om0.29997_As2.10000_mva0.00000_mvb0.00000_mvc0.00000_h0.70000_Ode0.69995/1024b512/box5/output_eb_5000_s4/seed0/'

    
######### local
#stats_dir = '/Users/jia/Dropbox/weaklensing/PDF/'
#ebcov_dir = stats_dir+'box5/output_eb_5000_s4/seed0/'

eb_dir = stats_dir+'stats_avg/output_eb_5000_s4/'
eb1k_dir = stats_dir+'stats_avg_1k/output_eb_5000_s4/'

#####################################
##### initiate avg statistics #######
#####################################

###### PS shape:(15, 101, 20)
psI = array( [load(eb_dir+'ALL_igalXigal_z{0}_z{1}_{2}.npy'.format(z_arr[i],z_arr[j],Nk))
              for i in range(Nz) for j in range(i,Nz)])
#psI = array( [load(eb_dir+'ALL_igalXigal_z{0}_z{0}_{1}.npy'.format(iz,Nk)) for iz in z_arr])
#psI1ks = array( [[load(eb1k_dir+'ALL_igalXigal_z{0}_z{0}_1k{1}.npy'.format(iz,ik)) for iz in z_arr] 
                 #for ik in range(10)])
#psN = array( [load(eb_dir+'ALL_galXgal_z{0}_z{0}_{1}.npy'.format(iz,Nk)) for iz in z_arr])

##### 1d PDF shape:(5, 101, 27)
#pdf1dN = array( [load(eb_dir+'ALL_gal_pdf_z{0}_sg1.0_{1}.npy'.format(iz,Nk)) for iz in z_arr])
#pdf1dN1ks = array( [[load(eb1k_dir+'ALL_gal_pdf_z{0}_sg1.0_1k{1}.npy'.format(iz,ik)) for iz in z_arr] 
                 #for ik in range(10)])

#### 2d PDF shape:(10, 101, 27, 27)
pdf2dN = array( [load(eb_dir+'ALL_galXgal_2dpdf_z{0}_z{1}_sg1.0_{2}.npy'.format(z_arr[i],z_arr[j],Nk)) 
                for i in range(Nz) for j in range(i+1,Nz)])
#pdf2dN1ks = array( [[load(eb1k_dir+'ALL_galXgal_2dpdf_z{0}_z{1}_sg1.0_1k{2}.npy'.format(z_arr[i],z_arr[j], ik)) 
                #for i in range(Nz) for j in range(i+1,Nz)] for ik in range(10)])

########## test collapsed 1d PDF from 2d shape:(5, 101, 27)
pdf1dN = array([sum(pdf2dN[i],axis=-1) for i in [0,4,7,9] ] + [sum(pdf2dN[-1],axis=-2)])

#####################################
###### covariances stats ############
#####################################

##### PS shape:(101,100)
psI_flat = swapaxes(psI,0,1).reshape(101,-1) 
#psI1k_flat = array([swapaxes(ips,0,1).reshape(101,-1) for ips in psI1ks])

psN_cov = swapaxes(array( [load(ebcov_dir+'ALL_galXgal_z{0}_z{1}.npy'.format(z_arr[i],z_arr[j]))
                           for i in range(Nz) for j in range(i,Nz)]),0,1).reshape(10000,-1)
covpsN = cov(psN_cov,rowvar=0)*12.25/2e4
covIpsN = mat(covpsN).I

###### PDF 1D
idxt=where(pdf1dN[:,5]>Nmin)#range(10, 20)#

#pdf1dN_flat= swapaxes(pdf1dN[idxt[0],:,idxt[1]],0,1).reshape(101,-1) 
##pdf1dN1k_flat = array([swapaxes(ips[idxt[0],:,idxt[1]],0,1).reshape(101,-1) for ips in pdf1dN1ks])

#pdf1dN_cov = swapaxes(array( [load(ebcov_dir+'ALL_gal_pdf_z{0}_sg1.0.npy'.format(iz)) for iz in z_arr])[idxt[0],:,idxt[1]],0,1).reshape(10000,-1)
#covpdf1dN = cov(pdf1dN_cov,rowvar=0)*12.25/2e4
#covIpdf1dN = mat(covpdf1dN).I

###### PDF 2D
idxt2=where(pdf2dN[:,5]>Nmin)

pdf2dN_flat= swapaxes(pdf2dN,0,1)[:,idxt2[0],idxt2[1],idxt2[2]]
#pdf2dN1k_flat= array([swapaxes(ips,0,1)[:,idxt2[0],idxt2[1],idxt2[2]] for ips in pdf2dN1ks])

pdf2dN_cov = swapaxes(array( [load(ebcov_dir+'ALL_galXgal_2dpdf_z{0}_z{1}_sg1.0.npy'.format(z_arr[i],z_arr[j]))
                             for i in range(Nz) for j in range(i+1,Nz)]),0,1)[:,idxt2[0],idxt2[1],idxt2[2]].reshape(10000,-1)

covpdf2dN = cov(pdf2dN_cov,rowvar=0)*12.25/2e4
covIpdf2dN = mat(covpdf2dN).I

############### test collapsed 1d PDF from 2d
pdf1dN_cov = array([sum(load(ebcov_dir+'ALL_galXgal_2dpdf_z{0}_z{1}_sg1.0.npy'.format(z_arr[i],z_arr[i+1])),axis=-1) for i in range(4) ] + [sum(load(ebcov_dir+'ALL_galXgal_2dpdf_z2.0_z2.5_sg1.0.npy'),axis=-2)])[idxt[0],:,idxt[1]].T
covpdf1dN = cov(pdf1dN_cov,rowvar=0)*12.25/2e4
covIpdf1dN = mat(covpdf1dN).I

#####################################
###### build emulator ###############
#####################################

params = genfromtxt(stats_dir+'cosmo_params_all.txt',usecols=[2,3,4])
fidu_params = array([0.1,0.3,2.1])

######## pick the good cosmology, where std/P among 10 1k models is <1%, and remove the first cosmology, 0eV one
#psI1k_std = std(psI1ks,axis=0)
#frac_diff = psI1k_std/psI[:,1].reshape(Nz,1,20)
#idx_good = where(amax(mean(frac_diff,axis=-1),axis=0)<0.01)[0][1:] 

############## test 6/23, use different IC for building emulator ########
#np.random.seed(10027)
#idx10 = list(np.random.randint(0,10, 101))
#stats = [array([istats[idx10[i],i] for i in range(101)]) for istats in [psI1k_flat, pdf1dN1k_flat, pdf2dN1k_flat]]
######################################

#stats = [psI_flat, pdf1dN_flat, pdf2dN_flat]
obss = [psI_flat[1], pdf1dN_flat[1], pdf2dN_flat[1]]
covIs = [covIpsN, covIpdf1dN, covIpdf2dN]

emulators = [WLanalysis.buildInterpolator(array(istats)[1:], params[1:], function='GP') 
             for istats in [psI_flat, pdf1dN_flat, pdf2dN_flat]]

chisq = lambda obs, model, covI: float(mat(obs-model)*covI*mat(obs-model).T)

################# for brute force grid method #############
#def ichisq (param):
    #return [float(chisq(stats[m][1], emulators[m](param), covIs[m])) for m in range(2)]
###########################################################

############### batch emulator ##########
#stats_batch = [psI1k_flat, pdf1dN1k_flat]
#idx_batch = [list(x) for x in itertools.combinations(range(10), 2)]
#chisq_batch = lambda obs1, obs2, model1, model2, covI: float(mat(obs1-model1)*covI*mat(obs2-model2).T)
#emul_batch  = [[WLanalysis.buildInterpolator(istats[i][idx_good], params[idx_good]) for i in range(10)] for istats in stats_batch]#, pdf2dN1k_flat]] ## shape (3,10) emulators

#def ichisq_batch (param):
    #chi2_arr = array([mean(array([chisq_batch(stats_batch[i][idx[0]][1], stats_batch[i][idx[1]][1],
                    #emul_batch[i][idx[0]](param), emul_batch[i][idx[1]](param), covIs[i]) 
                    #for idx in idx_batch])) for i in range(2)])
    #return abs(chi2_arr)

#########
def lnprob(p,jjj):
    '''log likelihood of 
    '''
    diff = emulators[jjj](p)-obss[jjj]
    return float(-0.5*mat(diff)*covIs[jjj]*mat(diff).T)


pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

print Nk


nwalkers=272
ndim=3
np.random.seed(10025)
p0 = (array([ (rand(nwalkers, ndim) -0.5) * array([1, 0.3, 0.3]) + 1]) * fidu_params).reshape(-1,3)

#print 'PS'
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[0,], pool=pool)
#pos, prob, state = sampler.run_mcmc(p0, 100)
#sampler.reset()
#sampler.run_mcmc(pos, Nchain)
#save(stats_dir+'likelihood/MC_ps_%s%s.npy'%(Nk,testfn), sampler.flatchain)

print 'PDF 1D'
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[1,], pool=pool)
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
sampler.run_mcmc(pos, Nchain*10)
save(stats_dir+'likelihood/MC_pdf1d_%s%s.npy'%(Nk,testfn), sampler.flatchain)

#print 'PDF 2D'
##p0 = (array([ (rand(nwalkers, ndim) -0.5) * array([1, 0.3, 0.3]) + 1]) * fidu_params).reshape(-1,3)
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[2,], pool=pool)
#pos, prob, state = sampler.run_mcmc(p0, 100)
#sampler.reset()
#sampler.run_mcmc(pos, Nchain*10)
#save(stats_dir+'likelihood/MC_pdf2d_%s%s.npy'%(Nk,testfn), sampler.flatchain)


#################################
########### grid method #########
#################################

#param_range = [[0,0.35],[0.28, 0.32],[1.9,2.3]]
#param_arr = [linspace(param_range[i][0],param_range[i][1],Ngrid) for i in range(3)]
#param_list = array(meshgrid(param_arr[0],param_arr[1],param_arr[2],indexing='ij')).reshape(3,-1).T ## shape: Ngrid x (Ngrid+1) x (Ngrid+2), 3

#print Nk, Ngrid
#out=array(pool.map(ichisq, param_list))
##out=array(pool.map(ichisq_batch, param_list))#.reshape(Ngrid, Ngrid+1, Ngrid+2)
#print 'grids done'

#save(stats_dir+'likelihood/prob_{0}_N{1}_2stats_GPdefault'.format(Nk,Ngrid),out)

print 'done done done'

pool.close()
