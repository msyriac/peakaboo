from scipy import *
import numpy as np
import WLanalysis

z_arr = arange(0.5,3,0.5)
Nz = len(z_arr)

#####################################
######## set up folders #############
#####################################

######## stampede2
#stats_dir = '/scratch/02977/jialiu/peakaboo/'
#ebcov_dir = stats_dir+'stats/Om0.29997_As2.10000_mva0.00000_mvb0.00000_mvc0.00000_h0.70000_Ode0.69995/1024b512/box5/output_eb_5000_s4/seed0'

######### local
stats_dir = '/Users/jia/Dropbox/weaklensing/PDF/'
ebcov_dir = stats_dir+'box5/output_eb_5000_s4/seed0/'

eb_dir = stats_dir+'stats_avg/output_eb_5000_s4/'
eb1k_dir = stats_dir+'stats_avg_1k/output_eb_5000_s4/'

#####################################
##### initiate avg statistics #######
#####################################

###### PS
psI = array( [load(eb_dir+'ALL_igalXigal_z{0}_z{0}_10k.npy'.format(iz)) for iz in z_arr])
psN = array( [load(eb_dir+'ALL_galXgal_z{0}_z{0}_10k.npy'.format(iz)) for iz in z_arr])

psI5ka = array( [load(eb_dir+'ALL_igalXigal_z{0}_z{0}_5ka.npy'.format(iz)) for iz in z_arr])
psN5ka = array( [load(eb_dir+'ALL_galXgal_z{0}_z{0}_5ka.npy'.format(iz)) for iz in z_arr])
psI5kb = array( [load(eb_dir+'ALL_igalXigal_z{0}_z{0}_5kb.npy'.format(iz)) for iz in z_arr])
psN5kb = array( [load(eb_dir+'ALL_galXgal_z{0}_z{0}_5kb.npy'.format(iz)) for iz in z_arr])

psI1ks = array( [[load(eb1k_dir+'ALL_igalXigal_z{0}_z{0}_1k{1}.npy'.format(iz,ik)) for iz in z_arr] 
                 for ik in range(10)])

##### 1d PDF
pdf1dI = array([load(eb_dir+'ALL_igal_pdf_z{0}_sg1.0_10k.npy'.format(iz)) for iz in z_arr])
pdf1dN = array( [load(eb_dir+'ALL_gal_pdf_z{0}_sg1.0_10k.npy'.format(iz)) for iz in z_arr])

pdf1dI5ka = array( [load(eb_dir+'ALL_igal_pdf_z{0}_sg1.0_5ka.npy'.format(iz)) for iz in z_arr])
pdf1dN5ka = array( [load(eb_dir+'ALL_gal_pdf_z{0}_sg1.0_5ka.npy'.format(iz)) for iz in z_arr])
pdf1dI5kb = array( [load(eb_dir+'ALL_igal_pdf_z{0}_sg1.0_5kb.npy'.format(iz)) for iz in z_arr])
pdf1dN5kb = array( [load(eb_dir+'ALL_gal_pdf_z{0}_sg1.0_5kb.npy'.format(iz)) for iz in z_arr])

pdf1dI1ks = array( [[load(eb1k_dir+'ALL_igal_pdf_z{0}_sg1.0_1k{1}.npy'.format(iz,ik)) for iz in z_arr] 
                 for ik in range(10)])
pdf1dN1ks = array( [[load(eb1k_dir+'ALL_gal_pdf_z{0}_sg1.0_1k{1}.npy'.format(iz,ik)) for iz in z_arr] 
                 for ik in range(10)])

#### 2d PDF
pdf2dI = array( [load(eb_dir+'ALL_igalXigal_2dpdf_z{0}_z{1}_sg1.0_10k.npy'.format(z_arr[i],z_arr[j])) 
                 for i in range(Nz) for j in range(i+1,Nz)])
pdf2dN = array( [load(eb_dir+'ALL_galXgal_2dpdf_z{0}_z{1}_sg1.0_10k.npy'.format(z_arr[i],z_arr[j])) 
                 for i in range(Nz) for j in range(i+1,Nz)])

pdf2dI5ka = array( [load(eb_dir+'ALL_igalXigal_2dpdf_z{0}_z{1}_sg1.0_5ka.npy'.format(z_arr[i],z_arr[j])) 
                 for i in range(Nz) for j in range(i+1,Nz)])
pdf2dN5ka = array( [load(eb_dir+'ALL_galXgal_2dpdf_z{0}_z{1}_sg1.0_5ka.npy'.format(z_arr[i],z_arr[j])) 
                 for i in range(Nz) for j in range(i+1,Nz)])
pdf2dI5kb = array( [load(eb_dir+'ALL_igalXigal_2dpdf_z{0}_z{1}_sg1.0_5kb.npy'.format(z_arr[i],z_arr[j])) 
                 for i in range(Nz) for j in range(i+1,Nz)])
pdf2dN5kb = array( [load(eb_dir+'ALL_galXgal_2dpdf_z{0}_z{1}_sg1.0_5kb.npy'.format(z_arr[i],z_arr[j])) 
                 for i in range(Nz) for j in range(i+1,Nz)])

pdf2dI1ks = array( [[load(eb1k_dir+'ALL_igalXigal_2dpdf_z{0}_z{1}_sg1.0_1k{2}.npy'.format(z_arr[i],z_arr[j], ik)) 
                 for i in range(Nz) for j in range(i+1,Nz)] for ik in range(10)])
pdf2dN1ks = array( [[load(eb1k_dir+'ALL_galXgal_2dpdf_z{0}_z{1}_sg1.0_1k{2}.npy'.format(z_arr[i],z_arr[j], ik)) 
                 for i in range(Nz) for j in range(i+1,Nz)] for ik in range(10)])

#####################################
###### covariances stats ############
#####################################

##### PS

psI_flat, psI5ka_flat, psI5kb_flat = [swapaxes(ips,0,1).reshape(101,-1) for ips in [psI, psI5ka, psI5kb]]
psI1k_flats = [swapaxes(ips,0,1).reshape(101,-1) for ips in psI1ks]

psI_cov = swapaxes(array( [load(ebcov_dir+'ALL_igalXigal_z{0}_z{0}.npy'.format(iz)) for iz in z_arr]),0,1).reshape(10000,-1)
psN_cov = swapaxes(array( [load(ebcov_dir+'ALL_galXgal_z{0}_z{0}.npy'.format(iz)) for iz in z_arr]),0,1).reshape(10000,-1)

covpsI = cov(psI_cov,rowvar=0)*12.25/2e4
covpsN = cov(psN_cov,rowvar=0)*12.25/2e4

covIpsI = mat(covpsI).I
covIpsN = mat(covpsN).I

###### PDF 1D
idxt=where(pdf1dN[1]>5)#range(10, 20)#range(10,30)

pdf1dI_flats = [swapaxes(ips[:,:,idxt],0,1).reshape(101,-1) for ips in [pdf1dI, pdf1dI5ka, pdf1dI5kb]]
pdf1dN_flats= [swapaxes(ips[:,:,idxt],0,1).reshape(101,-1) for ips in [pdf1dN, pdf1dN5ka, pdf1dN5kb]]
#pdf1dI1k_flats = [swapaxes(ips[:,:,idxt],0,1).reshape(101,-1) for ips in pdf1dI1ks]
#pdf1dN1k_flats = [swapaxes(ips[:,:,idxt],0,1).reshape(101,-1) for ips in pdf1dN1ks]

pdf1dN_cov = swapaxes(array( [load(ebcov_dir+'ALL_gal_pdf_z{0}_sg1.0.npy'.format(iz))[:,idxt] for iz in z_arr]),0,1).reshape(10000,-1)
covpdf1dN = cov(pdf1dN_cov,rowvar=0)*12.25/2e4
covIpdf1dN = mat(covpdf1dN).I

pdf1dI_cov = swapaxes(array( [load(ebcov_dir+'ALL_igal_pdf_z{0}_sg1.0.npy'.format(iz))[:,idxt] for iz in z_arr]),0,1).reshape(10000,-1)
covpdf1dI = cov(pdf1dI_cov,rowvar=0)*12.25/2e4
covIpdf1dI = mat(covpdf1dI).I

###### PDF 2D

idxt2=where(pdf2dN[:,1]>5)
pdf2dN_flats= array([swapaxes(ips,0,1)[:,idxt2[0],idxt2[1],idxt2[2]] for ips in [pdf2dN, pdf2dN5ka, pdf2dN5kb]])

pdf2dN_cov = swapaxes(array( [load(ebcov_dir+'ALL_galXgal_2dpdf_z{0}_z{1}_sg1.0.npy'.format(z_arr[i],z_arr[j]))
                             for i in range(Nz) for j in range(i+1,Nz)]),0,1)[:,idxt2[0],idxt2[1],idxt2[2]]

covpdf2dN = cov(pdf2dN_cov,rowvar=0)*12.25/2e4
covIpdf2dN = mat(covpdf2dN).I

