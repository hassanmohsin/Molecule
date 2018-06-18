
# coding: utf-8

# van der Waals radii are taken from A. Bondi, J. Phys. Chem., 68, 441 - 452, 1964, <br>
# except the value for H, which is taken from R.S. Rowland & R. Taylor, J.Phys.Chem., 100, 7384 - 7391, 1996. <br>
# Radii that are not available in either of these publications have RvdW = 2.00 <br>
# The radii for Ions (Na, K, Cl, Ca, Mg, and Cs are based on the CHARMM27 Rmin/2 parameters for (SOD, POT, CLA, CAL, MG, CES) by default.

from __future__ import print_function, absolute_import
from collections import OrderedDict
import numpy as np
from scipy import spatial
import pybel
import os
import glob
from tqdm import *

# Molecule class that assigns property of atom to a single voxel
class Molecule1:
    mol = None
    coords = []
    charges = []
    elements = []
    numAtoms = 0
    filename = ""
    _dir_name = ""
    
    _element_radii = {
        'Ac': 2.0,
        'Ag': 1.72,
        'Al': 2.0,
        'Am': 2.0,
        'Ar': 1.88,
        'As': 1.85,
        'At': 2.0,
        'Au': 1.66,
        'B': 2.0,
        'Ba': 2.0,
        'Be': 2.0,
        'Bh': 2.0,
        'Bi': 2.0,
        'Bk': 2.0,
        'Br': 1.85,
        'C': 1.7,
        'Ca': 1.37,
        'Cd': 1.58,
        'Ce': 2.0,
        'Cf': 2.0,
        'Cl': 2.27,
        'Cm': 2.0,
        'Co': 2.0,
        'Cr': 2.0,
        'Cs': 2.1,
        'Cu': 1.4,
        'Db': 2.0,
        'Ds': 2.0,
        'Dy': 2.0,
        'Er': 2.0,
        'Es': 2.0,
        'Eu': 2.0,
        'F': 1.47,
        'Fe': 2.0,
        'Fm': 2.0,
        'Fr': 2.0,
        'Ga': 1.07,
        'Gd': 2.0,
        'Ge': 2.0,
        'H': 1.2,
        'He': 1.4,
        'Hf': 2.0,
        'Hg': 1.55,
        'Ho': 2.0,
        'Hs': 2.0,
        'I': 1.98,
        'In': 1.93,
        'Ir': 2.0,
        'K': 1.76,
        'Kr': 2.02,
        'La': 2.0,
        'Li': 1.82,
        'Lr': 2.0,
        'Lu': 2.0,
        'Md': 2.0,
        'Mg': 1.18,
        'Mn': 2.0,
        'Mo': 2.0,
        'Mt': 2.0,
        'N': 1.55,
        'Na': 1.36,
        'Nb': 2.0,
        'Nd': 2.0,
        'Ne': 1.54,
        'Ni': 1.63,
        'No': 2.0,
        'Np': 2.0,
        'O': 1.52,
        'Os': 2.0,
        'P': 1.8,
        'Pa': 2.0,
        'Pb': 2.02,
        'Pd': 1.63,
        'Pm': 2.0,
        'Po': 2.0,
        'Pr': 2.0,
        'Pt': 1.72,
        'Pu': 2.0,
        'Ra': 2.0,
        'Rb': 2.0,
        'Re': 2.0,
        'Rf': 2.0,
        'Rg': 2.0,
        'Rh': 2.0,
        'Rn': 2.0,
        'Ru': 2.0,
        'S': 1.8,
        'Sb': 2.0,
        'Sc': 2.0,
        'Se': 1.9,
        'Sg': 2.0,
        'Si': 2.1,
        'Sm': 2.0,
        'Sn': 2.17,
        'Sr': 2.0,
        'Ta': 2.0,
        'Tb': 2.0,
        'Tc': 2.0,
        'Te': 2.06,
        'Th': 2.0,
        'Ti': 2.0,
        'Tl': 1.96,
        'Tm': 2.0,
        'U': 1.86,
        'V': 2.0,
        'W': 2.0,
        'X': 1.5,
        'Xe': 2.16,
        'Y': 2.0,
        'Yb': 2.0,
        'Zn': 1.39,
        'Zr': 2.0
    }
    
    _element_mapping = {
        'H': 'H',
        'HS': 'H',
        'HD': 'H',
        'A': 'C', 
        'C': 'C',
        'N': 'N',
        'NA': 'N',
        'NS': 'N',
        'O': 'O',
        'OA': 'O',
        'OS': 'O',
        'F': 'F',
        'Mg': 'Mg',
        'MG': 'Mg',
        'P': 'P',
        'S': 'S',
        'SA': 'S',
        'Cl': 'Cl',
        'CL': 'Cl',
        'Ca': 'Ca',
        'CA': 'Ca',
        'Fe': 'Fe',
        'FE': 'Fe',
        'Zn': 'Zn',
        'ZN': 'Zn',
        'BR': 'Br',
        'Br': 'Br',
        'I': 'I',
        'MN': 'Mn'
    }
    
    def __init__(self, file):
        self.filename = file
        self._read_file()
        self.mol = next(pybel.readfile('pdbqt', file))
    
    def _read_file(self):
        with open(self.filename, 'r') as f:
            content = f.readlines()
        
        # Split lines for space character
        content = [s.split() for s in content]
        # Choose only those that starts with "ATOM"
        content = [line for line in content if line[0]=="ATOM"]
        # Get the attributes
        self.coords = np.array([line[-7:-4] for line in content], dtype=np.float32)
        self.charges = np.array([line[-2] for line in content], dtype=np.float32)
        self.elements = np.array([line[-1] for line in content], dtype=object)
        self.numAtoms = self.elements.shape[0]
        
    
    def getVoxelDescriptors(self, side=1):
        voxel_side = side # in Angstorm
        
        # Get the channels for each of the properties
        elements = np.array([e.upper() for e in self.elements])
        properties = OrderedDict()
        _prop_order = ['hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor', 'positive_ionizable',
                       'negative_ionizable', 'metal', 'occupancies']

        properties['hydrophobic'] = (self.elements == 'C') | (self.elements == 'A')
        properties['aromatic'] = self.elements == 'A'
        properties['hbond_acceptor'] = (self.elements == 'NA') | (self.elements == 'NS') | (self.elements == 'OA') | (self.elements == 'OS') | (self.elements == 'SA')
        #properties['hbond_acceptor'] = np.array([a.OBAtom.IsHbondAcceptor() for a in self.mol.atoms], dtype=np.bool)
        properties['hbond_donor'] = np.array([a.OBAtom.IsHbondDonor() for a in self.mol.atoms], dtype=np.bool)
        properties['positive_ionizable'] = self.charges > 0.0
        properties['negative_ionizable'] = self.charges < 0.0
        properties['metal'] = (self.elements == 'MG') | (self.elements == 'ZN') | (self.elements == 'MN') | (self.elements == 'CA') | (self.elements == 'FE')
        properties['occupancies'] = (self.elements != 'H') & (self.elements != 'HS') & (self.elements != 'HD')
        
        channels = np.zeros((len(self.elements), len(properties)), dtype=bool)
        for i, p in enumerate(_prop_order):
            channels[:, i] = properties[p]
        
        # Now get the Van Dar Wals redii for each of the atoms
        vdw_radii = np.array([self._element_radii[self._element_mapping[elm]] 
                               for elm in self.elements], dtype=np.float32)
        
        # Multiply the vdw radii with the channel. False's will be zeros and True's will be the vdw radii
        channels = vdw_radii[:, np.newaxis] * channels.astype(np.float32)
            
        # Get the bounding box for the molecule
        max_coord = np.max(self.coords, axis=0) # np.squeeze?
        min_coord = np.min(self.coords, axis=0) # np.squeeze?
        
        # Calculate the number of voxels required
        N = np.ceil((max_coord - min_coord) / voxel_side).astype(int) + 1
        
        # Get the centers of each descriptors
        xrange = [min_coord[0] + voxel_side * x for x in range(0, N[0])]
        yrange = [min_coord[1] + voxel_side * x for x in range(0, N[1])]
        zrange = [min_coord[2] + voxel_side * x for x in range(0, N[2])]
        centers = np.zeros((N[0], N[1], N[2], 3))
        
        for i, x in enumerate(xrange):
            for j, y in enumerate(yrange):
                for k, z in enumerate(zrange):
                    centers[i, j, k, :] = np.array([x, y, z])
        
        centers = centers.reshape((-1, 3))
        features = np.zeros((len(centers), channels.shape[1]), dtype=np.float32)
        #features = np.zeros((len(centers)), dtype=np.float32)

        for i in range(self.numAtoms):
            # Get the atom coordinates
            atom_coordinates = self.coords[i]
            
            # Get the closest voxel
            c_voxel_id = spatial.distance.cdist(atom_coordinates.reshape((-1, 3)), centers).argmin()
            c_voxel = centers[c_voxel_id]
            
            # Calculate the potential
            voxel_distance = np.linalg.norm(atom_coordinates - c_voxel)
            x = channels[i] / voxel_distance
            #x = self._element_radii[self._element_mapping[self.elements[i]]] / voxel_distance
            n = 1.0 - np.exp(-np.power(x, 12))
            features[c_voxel_id] = n

            #break
            
        return features.reshape((N[0], N[1], N[2], -1))


# Molecule class that assigns property of atom to a single voxel and it's 8 neighbors
class Molecule2:
    mol = None
    coords = []
    charges = []
    elements = []
    numAtoms = 0
    filename = ""
    _dir_name = ""
    
    _element_radii = {
        'Ac': 2.0,
        'Ag': 1.72,
        'Al': 2.0,
        'Am': 2.0,
        'Ar': 1.88,
        'As': 1.85,
        'At': 2.0,
        'Au': 1.66,
        'B': 2.0,
        'Ba': 2.0,
        'Be': 2.0,
        'Bh': 2.0,
        'Bi': 2.0,
        'Bk': 2.0,
        'Br': 1.85,
        'C': 1.7,
        'Ca': 1.37,
        'Cd': 1.58,
        'Ce': 2.0,
        'Cf': 2.0,
        'Cl': 2.27,
        'Cm': 2.0,
        'Co': 2.0,
        'Cr': 2.0,
        'Cs': 2.1,
        'Cu': 1.4,
        'Db': 2.0,
        'Ds': 2.0,
        'Dy': 2.0,
        'Er': 2.0,
        'Es': 2.0,
        'Eu': 2.0,
        'F': 1.47,
        'Fe': 2.0,
        'Fm': 2.0,
        'Fr': 2.0,
        'Ga': 1.07,
        'Gd': 2.0,
        'Ge': 2.0,
        'H': 1.2,
        'He': 1.4,
        'Hf': 2.0,
        'Hg': 1.55,
        'Ho': 2.0,
        'Hs': 2.0,
        'I': 1.98,
        'In': 1.93,
        'Ir': 2.0,
        'K': 1.76,
        'Kr': 2.02,
        'La': 2.0,
        'Li': 1.82,
        'Lr': 2.0,
        'Lu': 2.0,
        'Md': 2.0,
        'Mg': 1.18,
        'Mn': 2.0,
        'Mo': 2.0,
        'Mt': 2.0,
        'N': 1.55,
        'Na': 1.36,
        'Nb': 2.0,
        'Nd': 2.0,
        'Ne': 1.54,
        'Ni': 1.63,
        'No': 2.0,
        'Np': 2.0,
        'O': 1.52,
        'Os': 2.0,
        'P': 1.8,
        'Pa': 2.0,
        'Pb': 2.02,
        'Pd': 1.63,
        'Pm': 2.0,
        'Po': 2.0,
        'Pr': 2.0,
        'Pt': 1.72,
        'Pu': 2.0,
        'Ra': 2.0,
        'Rb': 2.0,
        'Re': 2.0,
        'Rf': 2.0,
        'Rg': 2.0,
        'Rh': 2.0,
        'Rn': 2.0,
        'Ru': 2.0,
        'S': 1.8,
        'Sb': 2.0,
        'Sc': 2.0,
        'Se': 1.9,
        'Sg': 2.0,
        'Si': 2.1,
        'Sm': 2.0,
        'Sn': 2.17,
        'Sr': 2.0,
        'Ta': 2.0,
        'Tb': 2.0,
        'Tc': 2.0,
        'Te': 2.06,
        'Th': 2.0,
        'Ti': 2.0,
        'Tl': 1.96,
        'Tm': 2.0,
        'U': 1.86,
        'V': 2.0,
        'W': 2.0,
        'X': 1.5,
        'Xe': 2.16,
        'Y': 2.0,
        'Yb': 2.0,
        'Zn': 1.39,
        'Zr': 2.0
    }
    
    _element_mapping = {
        'H': 'H',
        'HS': 'H',
        'HD': 'H',
        'A': 'C', 
        'C': 'C',
        'N': 'N',
        'NA': 'N',
        'NS': 'N',
        'O': 'O',
        'OA': 'O',
        'OS': 'O',
        'F': 'F',
        'Mg': 'Mg',
        'MG': 'Mg',
        'P': 'P',
        'S': 'S',
        'SA': 'S',
        'Cl': 'Cl',
        'CL': 'Cl',
        'Ca': 'Ca',
        'CA': 'Ca',
        'Fe': 'Fe',
        'FE': 'Fe',
        'Zn': 'Zn',
        'ZN': 'Zn',
        'BR': 'Br',
        'Br': 'Br',
        'I': 'I',
        'MN': 'Mn'
    }
    
    def __init__(self, file):
        self.filename = file
        self._read_file()
        self.mol = next(pybel.readfile('pdbqt', file))
    
    def _read_file(self):
        with open(self.filename, 'r') as f:
            content = f.readlines()
        
        # Split lines for space character
        content = [s.split() for s in content]
        # Choose only those that starts with "ATOM"
        content = [line for line in content if line[0]=="ATOM"]
        # Get the attributes
        self.coords = np.array([line[-7:-4] for line in content], dtype=np.float32)
        self.charges = np.array([line[-2] for line in content], dtype=np.float32)
        self.elements = np.array([line[-1] for line in content], dtype=object)
        self.numAtoms = self.elements.shape[0]
        
    
    def getVoxelDescriptors(self, side=1):        
        voxel_side = side # in Angstorm
        
        # Get the channels for each of the properties
        elements = np.array([e.upper() for e in self.elements])
        properties = OrderedDict()
        _prop_order = ['hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor', 'positive_ionizable',
                       'negative_ionizable', 'metal', 'occupancies']

        properties['hydrophobic'] = (self.elements == 'C') | (self.elements == 'A')
        properties['aromatic'] = self.elements == 'A'
        properties['hbond_acceptor'] = (self.elements == 'NA') | (self.elements == 'NS') | (self.elements == 'OA') | (self.elements == 'OS') | (self.elements == 'SA')
        #properties['hbond_acceptor'] = np.array([a.OBAtom.IsHbondAcceptor() for a in self.mol.atoms], dtype=np.bool)
        properties['hbond_donor'] = np.array([a.OBAtom.IsHbondDonor() for a in self.mol.atoms], dtype=np.bool)
        properties['positive_ionizable'] = self.charges > 0.0
        properties['negative_ionizable'] = self.charges < 0.0
        properties['metal'] = (self.elements == 'MG') | (self.elements == 'ZN') | (self.elements == 'MN') | (self.elements == 'CA') | (self.elements == 'FE')
        properties['occupancies'] = (self.elements != 'H') & (self.elements != 'HS') & (self.elements != 'HD')
        
        channels = np.zeros((len(self.elements), len(properties)), dtype=bool)
        for i, p in enumerate(_prop_order):
            channels[:, i] = properties[p]
        
        # Now get the Van Dar Wals redii for each of the atoms
        vdw_radii = np.array([self._element_radii[self._element_mapping[elm]] 
                               for elm in self.elements], dtype=np.float32)
        
        # Multiply the vdw radii with the channel. False's will be zeros and True's will be the vdw radii
        channels = vdw_radii[:, np.newaxis] * channels.astype(np.float32)
            
        # Get the bounding box for the molecule
        max_coord = np.max(self.coords, axis=0) # np.squeeze?
        min_coord = np.min(self.coords, axis=0) # np.squeeze?
        
        # Calculate the number of voxels required
        N = np.ceil((max_coord - min_coord) / voxel_side).astype(int) + 1
        
        # Get the centers of each descriptors
        xrange = [min_coord[0] + voxel_side * x for x in range(0, N[0])]
        yrange = [min_coord[1] + voxel_side * x for x in range(0, N[1])]
        zrange = [min_coord[2] + voxel_side * x for x in range(0, N[2])]
        centers = np.zeros((N[0], N[1], N[2], 3))
        
        for i, x in enumerate(xrange):
            for j, y in enumerate(yrange):
                for k, z in enumerate(zrange):
                    centers[i, j, k, :] = np.array([x, y, z])
        
        centers = centers.reshape((-1, 3))
        features = np.zeros((len(centers), channels.shape[1]), dtype=np.float32)
        #features = np.zeros((len(centers)), dtype=np.float32)

        for i in range(self.numAtoms):
            # Get the atom coordinates
            atom_coordinates = self.coords[i]
            
            # Get the closest voxel and it's 8 neighbors ids and distances
            voxel_distances = spatial.distance.cdist(atom_coordinates.reshape((-1, 3)), centers).reshape(-1)
            c_voxel_ids = voxel_distances.argsort()[:9]
            c_voxel_dist = np.sort(voxel_distances)[:9]
            
            
            # Calculate the potential
            #voxel_distance = np.linalg.norm(atom_coordinates - c_voxel)
            x = channels[i] / c_voxel_dist.reshape(-1)[:, np.newaxis]
            #x = self._element_radii[self._element_mapping[self.elements[i]]] / voxel_distance
            n = 1.0 - np.exp(-np.power(x, 12))
            
            # Get the maximum and assign
            max_feat = np.maximum(features[c_voxel_ids], n)
            
            features[c_voxel_ids] = n

        return features.reshape((N[0], N[1], N[2], -1))