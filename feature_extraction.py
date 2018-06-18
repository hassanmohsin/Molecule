
import multiprocessing
from multiprocessing import Manager, current_process
from Molecule import Molecule1, Molecule2
import time
import glob
import numpy as np
import pybel
from collections import OrderedDict
from scipy import spatial
import os
import pickle

files = glob.glob("/home/PDBbind/pdbbind_2016/refined-set-2016/*/*_pocket.pdbqt")
print("Number of files {}".format(len(files)))

def get_feat(file):
    _, f = os.path.split(file)
    _id = f[:4]
    pid = current_process().name
    #print("Procesing {} from {}".format(file, pid))
    try:
        mol = Molecule2(file)
        feat = mol.getVoxelDescriptors(side=3)
    except:
        return None
    return _id, feat

process_count = multiprocessing.cpu_count()
p = multiprocessing.Pool(process_count)
results = p.map(get_feat, files)


feature_dict = {}

for result in results:
    if result == None:
        continue
    _id, _feat = result
    feature_dict[_id] = _feat


with open('data/features_side_3_neigh_ref.pickle', 'wb') as f:
    pickle.dump(feature_dict, f)

