import pandas as pd
from st_rate import structures as struct
from st_rate import cluster as clst

path = '/xspace/db4271/ST_Rate/data' # main path to structure data
mytop = f'{path}/prmtop/triad_NOthf_bent_GR.prmtop'

# this data set contain 100,000 snapshots
mytraj_dataset = f'{path}/traj/triad_dataset.nc'

# get atom type or atomformat for the structure 
structure_file=f'{path}/structure/triad_sample.xyz'
atomformat_xyz= struct.get_atomformat(structure_file,skip_row=2) # newest version of st_rate have default skip_row=2

# update this pdb format loader to struct.get_atomformat
structure_file=f'{path}/structure/triad_sample.pdb'
file_pdb=pd.read_table(structure_file, header=None,delim_whitespace=True) 
atomformat_pdb=file_pdb[2]

# save triad_dataset.nc to file 



#This is for saving structures into thousand of folders each with thousands of data
###################################################################################################
# Molecule Prop
total_atoms=207
resname = "TRI" # define resname for the pdbfile 
case='triad'
system=case+'_'


###########
################################################################################################3

# # STEP 1
#define naming convention
for i in range(0,100,1):    
    filepath = f'/xspace1/projects/CT_Landscape/data/triad_durf_dataset/triad_durf_dataset_{i}/'
    naming = system
    j = (i * 1000) 
    k = (j + 1000)
    index_list_full = pd.Series(np.arange(j,k,1))
    cc_index = index_list_full
    clst.save_cluster_centers(cc_index, 
                              naming, 
                              mytop, mytraj_dataset, 
                              atomformat_xyz, atomformat_pdb,
                              resname,total_atoms, 
                              filepath,
                              charge=0,
                              multiplicity=1,
                              notes = 'Triad, Carotenoid-Porphyrin-C60' )
