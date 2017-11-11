# peakaboo

A typical workflow would look like the following.

## Create CMB lensing reconstructions 

```
my_mpi_submitter --numcores=100 "python -W ignore bin/recon.py cosmology_dir_name tt_3000_s4 reconstruction_liu_TT_3000 experiment_s4 TT default_1d_bin"
```

Here, `my_mpi_submitter --numcores=100` is a schematic representation of whatever script you use to submit parallel MPI jobs. We have 1000 sims by default, and here this is just representing a submission that asks for 100 cores, each of which will work on 10 sims in parallel.

In detail:

1. `-W ignore` suppresses some annoying warnings that clog up the output
2. `bin/recon.py` is the reconstruction script
3. `cosmology_dir_name` is the name of the directory for the cosmology you are interested in. Your directory structure should be such that 1000 `input_data+cosmology_dir_name+"\Maps11000\WLConv*fits"` files exist.


