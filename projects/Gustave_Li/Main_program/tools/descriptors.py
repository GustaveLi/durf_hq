import mdtraj as md
import MDAnalysis as mda
import numpy as np
import pandas as pd

class Descriptors():
    """
    Descriptors class: contain attributes for triad descriptor calculation
    
    Input: trajectory file & topology file
    
    Output: descriptor data set
    """
    
    def __init__(self, traj_file, top_file):
        self.top_file = top_file
        self.traj = md.load(traj_file, top=top_file)
        self.u = mda.Universe(top_file, traj_file)
        
    def index_finder(self, atom_name_list):
        """
        Convert the atom name into atom index for selected atoms in the traj,
        prepare for calculation.

        Parameters
        ----------
        *atom_names : TYPE List or tuple
            DESCRIPTION. Atom names for conversion, could be one or multiple

        Returns
        -------
        idx : TYPE Numpy array, shape=(1, # of atom)
            DESCRIPTION. The index array for selected atoms
        """
        topology = self.traj.topology
        idx = np.array([atom.index for atom in topology.atoms if atom.name in \
               atom_name_list])
        return idx.reshape(1,len(idx))
    
    def eucdist(self, *atom_names):
        """
        Calculate euclidian distance between two atoms

        Parameters
        ----------
        *atom_names : TYPE String
            DESCRIPTION. Name of the two atoms for distance calculation

        Returns
        -------
        dist : TYPE Numpy array, shape=(# of frames, 1)
            DESCRIPTION. The distances are in angstrom
        """
        if len(atom_names)==2:
            idx_list = Descriptors.index_finder(self, atom_names)
            dist = md.compute_distances(self.traj, idx_list)
            return dist.reshape(len(dist),)
        else:
            print('Two atoms are expected for distance calculation')
            
    def angle(self, *atom_names):
        """
        Calculate angle between three atoms

        Parameters
        ----------
        *atom_names : TYPE String
            DESCRIPTION. Name of the three atoms for angle calculation

        Returns
        -------
        angle : TYPE Numpy array, shape=(# of frames, 1)
            DESCRIPTION. The angles are in radians

        """
        if len(atom_names)==3:
            idx_list = Descriptors.index_finder(self, atom_names)
            angle = md.compute_angles(self.traj, idx_list)
            return angle.reshape(len(angle),)
        else:
            print('Three atoms are expected for angle calculation')
            
    def dihedral(self, *atom_names):
        """
        Calculate dihedral between four atoms

        Parameters
        ----------
        *atom_names : TYPE String
            DESCRIPTION. Name of the four atoms for dihedral calculation

        Returns
        -------
        angle : TYPE 1D Numpy array
            DESCRIPTION. The absolute value of the dihedrals. The dihedrals are in radians

        """
        if len(atom_names)==4:
            idx_list = Descriptors.index_finder(self, atom_names)
            torsion = md.compute_dihedrals(self.traj, idx_list)
            return np.abs(torsion.reshape(len(torsion),))
        else:
            print('Four atoms are expected for dihedral calculation')
     
    def rmsd(self, ref_file=None, frame=0):
        """
        

        Parameters
        ----------
        ref_file : TYPE File path (string), optional
            DESCRIPTION. The file path for reference traj. The default is None.
        frame : TYPE Integer, optional
            DESCRIPTION. Reference frame number. The default is 0.

        Returns
        -------
        rmsd : TYPE 1-D Numpy array
            DESCRIPTION. The optimal root-mean-square deviations from the frame-th 
            conformation in reference to each of the conformations in target.

        """
        if ref_file == None:
            rmsd = md.rmsd(self.traj, self.traj, frame)
        else:
            ref = md.load(ref_file, top=self.top_file)
            rmsd = md.rmsd(self.traj, ref, frame)
        return rmsd
    
    def to_df(self, **descriptor):
        """
        Combine the descriptor(numpy array) to pandas dataframe

        Parameters
        ----------
        **descriptor : TYPE Dictionary
            DESCRIPTION. Key is the descriptor name, value is the numpy array

        Returns
        -------
        df : TYPE Pandas DataFrame

        """
        df = pd.DataFrame(descriptor)
        return df
        
        