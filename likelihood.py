from scipy import *
from emcee.utils import MPIPool 
import os

ALL_fn = genfromtxt('ALL_fn.txt',dtype='string')
fn_list = genfromtxt('cosmo_params_all.txt',usecols=1, dtype='string')
fnjia_list = genfromtxt('cosmo_params_all.txt',usecols=0, dtype='string')
stats_fn = lambda iALL, ieb, icosmo: '/scratch/02977/jialiu/peakaboo/stats/'+icosmo+'/1024b512/'+ieb+'/seed0/'+iALL
stats_dir = '/scratch/02977/jialiu/peakaboo/stats_avg/'

def compute_average(i):
    icosmo = fn_list[i]
    isavedir = stats_dir+icosmo+'/'
    os.system('mkdir -pv %s'%(icosmo))
    for ieb in ['output_eb_5000_s4', 'output_tt_3000_s4']:        
        for iALL in ALL_fn:
            idata = load(stats_fn(iALL, ieb, icosmo))
            j=0
            for k in (1000, 3000, 5000):
                save(isavedir+iALLL[:-4]+'_%sk_%s'%(k/1000,ieb), mean(idata[j:j+k],axis=0) )
                j+=k
            save(isavedir+iALLL[:-4]+'_10k_%s'%(ieb), mean(idata,axis=0) )

pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
    
pool.map(compute_average, range(len(fn_list))

print 'DONE-DONE-DONE'
