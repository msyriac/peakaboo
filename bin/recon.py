from __future__ import print_function
from peakaboo.utils import PeakabooPipeline
import argparse
from orphics.tools.mpi import MPI
import orphics.tools.io as io

# Parse command line
parser = argparse.ArgumentParser(description='Run lensing sims pipe.')
parser.add_argument("InpDir", type=str,help='Input Directory Name (not path, that\'s specified in ini)')
parser.add_argument("OutDir", type=str,help='Output Directory Name')
parser.add_argument("recon", type=str,help='Recon section name')
parser.add_argument("experiment", type=str,help='Name of experiment section')
parser.add_argument("estimator", type=str,help='TT/EB')
parser.add_argument("bin_section", type=str,help='bin_section')
parser.add_argument("-v", "--verbose", action='store_true',help='Talk more.')
parser.add_argument("-d", "--debug", action='store_true',help='Debug.')
parser.add_argument("-N", "--nmax",     type=int,  default=None,help="Limit to nmax sims.")
args = parser.parse_args()

experiment = args.experiment
Nmax = args.nmax
debug = args.debug


# Initialize pipeline
PathConfig = io.load_path_config()
pipe = PeakabooPipeline(args.estimator,PathConfig,args.InpDir,args.OutDir,args.nmax,
                        args.recon,args.experiment,
                        mpi_comm = MPI.COMM_WORLD,bin_section=args.bin_section,verbose = args.verbose)





# Loop through sims
for k,sim_id in enumerate(pipe.sim_ids):
    unlensed = pipe.get_unlensed(seed=sim_id)
    input_kappa = pipe.get_kappa(sim_id)
    lensed = pipe.downsample(pipe.get_lensed(unlensed,input_kappa))
        
    if debug: pipe.power_plotter(pipe.fc.iqu2teb(lensed,normalize=False),"lensed")
    beamed = pipe.beam(lensed)
    noise = pipe.get_noise(seed=sim_id)

    
    observed = beamed+noise
    lobserved = pipe.fc.iqu2teb(observed,normalize=False)
    if debug: pipe.power_plotter(lobserved/pipe.pdat.lbeam,"observed-deconvolved")
    
    if args.estimator=="TT":
        lT = lobserved
        lE = None
        lB = None
    else:
        lT = lobserved[0]
        lE = lobserved[1]
        lB = lobserved[2]
        
    kappa = pipe.qest(lT,lE,lB)

    
    pipe.save_kappa(kappa,sim_id)
    if pipe.rank==0 and (k+1)%1==0: pipe.logger.info( "Rank 0 done with "+str(k+1)+ " / "+str( len(pipe.sim_ids))+ " tasks.")
    
if pipe.rank==0: pipe.logger.info( "MPI Collecting...")
pipe.mpibox.get_stacks(verbose=False)
pipe.mpibox.get_stats(verbose=False)

if pipe.rank==0:
    pipe.dump(debug=debug)
