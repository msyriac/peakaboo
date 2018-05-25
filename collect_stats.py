from scipy import *
from emcee.utils import MPIPool 
import os

peaka_dir = '/home1/02977/jialiu/peakaboo/'
ALL_fn = genfromtxt(peaka_dir+'ALL_fn.txt',dtype='string')
fn_list = genfromtxt(peaka_dir+'cosmo_params_all.txt',usecols=1, dtype='string')
fnjia_list = genfromtxt(peaka_dir+'cosmo_params_all.txt',usecols=0, dtype='string')
stats_fn = lambda iALL, ieb, icosmo: '/scratch/02977/jialiu/peakaboo/stats/'+icosmo+'/1024b512/'+ieb+'/seed0/'+iALL
stats_dir = '/scratch/02977/jialiu/peakaboo/stats_avg/'
stats1k_dir = '/scratch/02977/jialiu/peakaboo/stats_avg_1k/'

def compute_average(i):
    icosmo = fn_list[i]
    print icosmo
    for ieb in ['output_eb_5000_s4', 'output_tt_3000_s4']:    
        isavedir = stats_dir+icosmo+'/'+ieb+'/'
        os.system('mkdir -pv %s'%(isavedir))
        for iALL in ALL_fn:
            
            idata = load(stats_fn(iALL, ieb, icosmo))
            save(isavedir+iALL[:-4]+'_10k', mean(idata,axis=0) )
            save(isavedir+iALL[:-4]+'_5ka', mean(idata[:5000],axis=0) )
            save(isavedir+iALL[:-4]+'_5kb', mean(idata[5000:],axis=0) )
            for k in range(10):
                save(isavedir+iALL[:-4]+'_1k%i'%(k), mean(idata[k*1000:(k+1)*1000],axis=0) )

def compute_average_bystats(iALL):
    print iALL
    for ieb in ['output_eb_5000_s4', 'output_tt_3000_s4']: 
        isavedir = stats_dir+ieb+'/'
        isavedir_1k = stats1k_dir+ieb+'/'
        os.system('mkdir -pv %s; mkdir -pv %s'%(isavedir, isavedir_1k))
        idatagen = lambda icosmo: laod(stats_fn(iALL, ieb, icosmo))
        all_idata = array(map(idatagen, fn_list))
        save(isavedir+iALL[:-4]+'_10k', mean(idata,axis=1) )
        save(isavedir+iALL[:-4]+'_5ka', mean(idata[:5000],axis=1) )
        save(isavedir+iALL[:-4]+'_5kb', mean(idata[5000:],axis=1) )
        for k in range(10):
            save(isavedir_1k+iALL[:-4]+'_1k%i'%(k), mean(idata[k*1000:(k+1)*1000],axis=1) )

        
pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
    
#pool.map(compute_average, range(len(fn_list)))
pool.map(compute_average_bystats, range(len(ALL_fn)))

pool.close()
print 'DONE-DONE-DONE'
