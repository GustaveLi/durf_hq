# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:37:45 2021

@author: gqmli
"""

import mdtraj as md
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.distances import distance_array

class Descriptor():
    """
    A combination of descriptors for the triad molecules.
    
    Attributes
    ----------
    __init__()
    com()
    """
    
    def __init__(self, top_file, traj_file):
        """

        Parameters
        ----------
        top_file : file path
            File path for topology file
        traj_file : file path
            File path for trajectory file

        Returns
        -------
        None.

        """
        self.traj = md.load(traj_file, top=top_file)
        self.u = mda.Universe(top_file, traj_file)
        self.C60 = self.u.atoms[:65] # Included the C60-aryl linkage
        self.porphyrin = self.u.atoms[65:121]
        self.carotenoid = self.u.atoms[:121] # Included the carotenoid-aryl linkage
    
    def com(self, specificity='entire'):
        """
        Obtain the center of mass for the traj or atom group

        Parameters
        ----------
        specificity : optional
            DESCRIPTION. User can specify which part of molecule do they want the com be computed (the entire molecule, C60, porphyrin or the carotenoid)
            The default is 'entire'.

        Returns
        -------
        Numpy array of the coordinates ( ,3)

        """
        if specificity=='entire':
            Com = md.compute_center_of_mass(self.traj)
        else:
            com_list = []
            self.u.trajectory[0]
            if specificity=='C60':
                x = self.C60
            elif specificity == 'porphyrin':
                x = self.porphyrin
            elif specificity == 'carotenoid':
                x = self.carotenoid
            for ts in self.u.trajectory:
                coordinate_com = x.center_of_mass()
                com_list.append(coordinate_com)
            Com = np.array(com_list)
        return Com
    
    def distance(self, pt1, pt2):
        """
        Calculate the distance between COM of different components in the triad molecule (C60, porphyrin, carotein)

        Parameters
        ----------
        pt1 : Component of the triad molecule
            DESCRIPTION. C60 / porphyrin / carotein
        pt2 : Component of the triad molecule
            DESCRIPTION. C60 / porphyrin / carotein

        Returns
        -------
        dist : Numpy array of the distances in each frame

        """
        center_of_mass1 = Descriptor.com(self,pt1)
        center_of_mass2 = Descriptor.com(self,pt2)
        dist_list = []
        for i in range(len(center_of_mass1)):
            D = distance_array(center_of_mass1[i], center_of_mass2[i])
            dist_list.append(D)
        dist = np.array(dist_list).reshape(len(center_of_mass1),)
        return dist
            