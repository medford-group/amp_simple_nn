import importlib
from collections import namedtuple, defaultdict

import numpy as np
import os
from ase import io
import json
from io import StringIO
import pickle

from ase.geometry.analysis import Analysis
from pymatgen.io.ase import AseAtomsAdaptor as adaptor
from ase.spacegroup import crystal
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from shutil import copyfile



def make_params_file(elements, etas, rs_s, g4_eta = 4, cutoff = 6.5,
                     g4_zeta=[1.0, 4.0], g4_gamma=[1, -1]):
    """
    makes a params file for simple_NN. This is the file containing
    the descriptors. This function makes g2 descriptos for the eta
    and rs values that are input, and g4 descriptors that are log
    spaced between 10 ** -5 and 10 ** -1. The number of these
    that are made is controlled by the `n_g4_eta` variable
    Parameters:
        elements (list):
            a list of elements for which you'd like to make params
            files for
        etas (list):
            the eta values you'd like to use for the descriptors
        rs_s (list):
            a list corresponding to `etas` that contains the rs
            values for each descriptor
        g4_eta (int or list):
            the number of g4 descriptors you'd like to use. if a
            list is passed in the values of the list will be used
            as eta values
        cutoff (float):
            the distance in angstroms at which you'd like to cut 
            off the descriptors
    
    returns:
        None
    
    """
    if type(g4_eta) == int:
        g4_eta = np.logspace(-4, -1, num = g4_eta)
    for element in elements:
        with open('params_{}'.format(element),'w') as f:
            # G2
            for species in range(1, len(elements) + 1):
                for eta, Rs in zip(etas, rs_s):
                    f.write('2 {} 0 {} {} {} 0.0\n'.format(species, cutoff,
                                                           np.round(eta, 6), Rs))
            # G4 
            for i in range(1, len(elements) + 1):
                n = i
                while True:
                    for eta in g4_eta:
                        for lamda in g4_gamma:
                            for zeta in g4_zeta:
                                f.write('4 {} {} {} {} {} {}\n'.format(i, n, cutoff,
                                                                         np.round(eta, 6),
                                                                         zeta, lamda))
                    n += 1
                    if n > len(elements):
                        break

def reorganize_simple_nn_derivative(image, dx_dict):
    """
    reorganizes the fingerprint derivatives from simplen_nn into
    amp format
    Parameters:
        image (ASE atoms object):
            the atoms object used to make the finerprint
        dx_dict (dict):
            a dictionary of the fingerprint derivatives from simple_nn
    """
    # TODO check for bugs
    d = defaultdict(list)
    sym_dict = defaultdict(list)
    syms = image.get_chemical_symbols()
    for i, sym in enumerate(syms):
        sym_dict[sym].append(i)
    # the structure is:
    # [elements][atom i][symetry function #][atom j][derivitive in direction]
    for element, full_arr in dx_dict.items():
        for i, arr_t in enumerate(full_arr):
            true_i = sym_dict[element][i]
            for sf in arr_t:
                for j, dir_arr in enumerate(sf):
                    for k, derivative in enumerate(dir_arr):
                        d[(true_i,element,j,syms[j],k)].append(derivative)
    zero_keys = []
    for key, derivatives in d.items():
        zero_check = [a == 0 for a in derivatives]
        if zero_check == [True] * len(derivatives):
            zero_keys.append(key)
    for key in zero_keys:
        del d[key]
    d = dict(d)
    return d

def reorganize_simple_nn_fp(image, x_dict):
    """
    reorganizes the fingerprints from simplen_nn into
    amp format
    Parameters:
        image (ASE atoms object):
            the atoms object used to make the finerprint
        x_dict (dict):
            a dictionary of the fingerprints from simple_nn
    """
    # TODO check for bugs
    # the structure is:
    # [elements][atom i][symetry function #][fp]
    fp_l = []
    sym_dict = defaultdict(list)
    syms = image.get_chemical_symbols()
    for i, sym in enumerate(syms):
        sym_dict[sym].append(i)
    for element, full_arr in x_dict.items():
        for i, fp  in enumerate(full_arr):
            true_i = sym_dict[i]
            fp_l.append((element,list(fp)))
    return fp_l

def get_hash(atoms):
    import hashlib
    """Creates a unique signature for a particular ASE atoms object.
    This is used to check whether an image has been seen before. This is just
    an md5 hash of a string representation of the atoms object.
    Parameters
    ----------
    atoms : ASE dict
        ASE atoms object.
    Returns
    -------
        Hash string key of 'atoms'.
    """
    string = str(atoms.pbc)
    try:
        flattened_cell = atoms.cell.array.flatten()
    except AttributeError:  # older ASE
        flattened_cell = atoms.cell.flatten()
    for number in flattened_cell:
        string += '%.15f' % number
    for number in atoms.get_atomic_numbers():
        string += '%3d' % number
    for number in atoms.get_positions().flatten():
        string += '%.15f' % number

    md5 = hashlib.md5(string.encode('utf-8'))
    hash = md5.hexdigest()
    return hash


def convert_simple_nn_fps(traj, delete_old=True):
    from multiprocessing import Pool
    # make the directories
    if not os.path.isdir('./amp-fingerprints.ampdb'):
        os.mkdir('./amp-fingerprints.ampdb')
    if not os.path.isdir('./amp-fingerprints.ampdb/loose'):
        os.mkdir('./amp-fingerprints.ampdb/loose')
    if not os.path.isdir('./amp-fingerprint-primes.ampdb'):
        os.mkdir('./amp-fingerprint-primes.ampdb')
    if not os.path.isdir('./amp-fingerprint-primes.ampdb/loose'):
        os.mkdir('amp-fingerprint-primes.ampdb/loose')
    # perform the reorganization
    """
    for i, image in enumerate(traj):
        pic = pickle.load(open('./data/data{}.pickle'.format(i + 1), 'rb'))
        im_hash = get_hash(image)
        x_list = reorganize_simple_nn_fp(image, pic['x'])
        pickle.dump(x_list, open('./amp-fingerprints.ampdb/loose/' + im_hash, 'wb'))
        del x_list  # free up memory just in case
        x_der_dict = reorganize_simple_nn_derivative(image, pic['dx'])
        pickle.dump(x_der_dict, open('./amp-fingerprint-primes.ampdb/loose/' + im_hash, 'wb'))
        del x_der_dict  # free up memory just in case
        if delete_old:  # in case disk space is an issue
            os.remove('./data/data{}.pickle'.format(i + 1))
    """
    with Pool(10) as p:
        l_trajs = list(enumerate(traj))
        p.map(reorganize, l_trajs)
    if delete_old:
        os.rmdir('./data')

def reorganize(inp, delete_old=True):
    i, image = inp
    pic = pickle.load(open('./data/data{}.pickle'.format(i + 1), 'rb'))
    im_hash = get_hash(image)
    x_list = reorganize_simple_nn_fp(image, pic['x'])
    pickle.dump(x_list, open('./amp-fingerprints.ampdb/loose/' + im_hash, 'wb'))
    del x_list  # free up memory just in case
    x_der_dict = reorganize_simple_nn_derivative(image, pic['dx'])
    pickle.dump(x_der_dict, open('./amp-fingerprint-primes.ampdb/loose/' + im_hash, 'wb'))
    del x_der_dict  # free up memory just in case
    if delete_old:  # in case disk space is an issue
        os.remove('./data/data{}.pickle'.format(i + 1))


class DummySimple_nn(object):
    """
    a dummy class to fool the simple_nn descriptor class into
    thinking it's attached to a simple_nn instance
    """
    def __init__(self, atom_types):
        self.inputs = {
            'generate_features': True,
            'preprocess': False,
            'train_model': True,
            'atom_types': atom_types}
        self.logfile = open('simple_nn_log', 'w')

def make_simple_nn_fps(traj, descriptors, clean_up_directory=True,
                       elements='all'):
    """
    generates descriptors using simple_nn. The files are stored in the
    ./data folder. These descriptors will be in the simple_nn form and
    not immediately useful for other programs
    Parameters:
        traj (list of ASE atoms objects):
            a list of the atoms you'd like to make descriptors for
        descriptors (tuple):
            a tuple containing (g2_etas, g2_rs_s, g4_etas, cutoff, g4_zetas, g4_gammas)
        clean_up_directory (bool):
            if set to True, the input files made by simple_nn will
            be deleted
    returns:
        None
    """
    from simple_nn.features.symmetry_function import Symmetry_function

    # handle inputs
    if type(traj) != list:
        traj = [traj]

    # clean up any previous runs
    if os.path.isdir('./data'):
        shutil.rmtree('./data')

    # set up the input files
    io.write('simple_nn_input_traj.traj',traj)
    with open('str_list', 'w') as f:
        f.write('simple_nn_input_traj.traj :') # simple_nn requires this file


    if elements == 'all':
        atom_types = []
        # TODO rewrite this
        for image in traj:
            atom_types += image.get_chemical_symbols()
            atom_types = list(set(atom_types))
    else:
        atom_types = elements

    make_params_file(atom_types, *descriptors)

    # build the descriptor object
    descriptor = Symmetry_function()
    params = {a:'params_{}'.format(a) for a in atom_types}

    descriptor.inputs = {'params': params, 
                         'refdata_format': 'traj', 
                         'compress_outcar': False,
                         'data_per_tfrecord': 150, 
                         'valid_rate': 0.1, 
                         'remain_pickle': False, 
                         'continue': False, 
                         'add_atom_idx': True, 
                         'num_parallel_calls': 5, 
                         'atomic_weights': {'type': None, 'params': {}}, 
                         'weight_modifier': {'type': None, 'params': {}}, 
                         'scale_type': 'minmax', 
                         'scale_scale': 1.0, 
                         'scale_rho': None}
    dummy_class = DummySimple_nn(atom_types=atom_types)
    descriptor.parent = dummy_class

    # generate the descriptors
    descriptor.generate()
    
    if clean_up_directory:
        # clean the folder of all the junk
        files = ['simple_nn_input_traj.traj', 'str_list',
                 'pickle_list', 'simple_nn_log']
        files += list(params.values())
        for file in files:
            os.remove(file)

def make_amp_descriptors_simple_nn(traj, g2_etas, g2_rs_s, g4_etas, g4_zetas, g4_gammas, cutoff):
    """
    uses simple_nn to make descriptors in the amp format.
    Only creates the same symmetry functions for each element
    for now.
    """
    c = cutoff
    #g2_etas = [a * cutoff for a in g2_etas]
    #g4_etas = [a * cutoff for a in g4_etas]
    make_simple_nn_fps(traj,
                       (g2_etas, g2_rs_s, g4_etas, 
                        cutoff, 
                        g4_zetas, g4_gammas),
                        clean_up_directory=True)
    convert_simple_nn_fps(traj, delete_old=True)

