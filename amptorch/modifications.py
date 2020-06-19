import copy
import os
import numpy as np
import torch
import scipy.sparse as sparse
from amptorch.data_preprocess import AtomsDataset
import pickle
from amptorch.utils import reorganize_simple_nn_derivative
from collections import OrderedDict
from amptorch.gaussian import make_symmetry_functions, SNN_Gaussian
import ase
from torch.utils.data import Dataset
from amptorch.utils import (
    make_simple_nn_fps,
    stored_fps,
    calculate_fingerprints_range,
    hash_images,
    get_hash,
)
from amp.utilities import hash_images as amp_hash
from amp.utilities import get_hash as get_amp_hash

class AtomsDataset_per_image(AtomsDataset):
    """
    deleted the normalization
    get loss per image
    """

    def __init__(self, images, descriptor, Gs, forcetraining, label, cores, delta_data=None, store_primes=False,
                 specific_atoms=False):
        super().__init__(images, descriptor, Gs, forcetraining, label, cores, delta_data, store_primes, specific_atoms)


    def preprocess_data(self):
        # TODO cleanup/optimize
        fingerprint_dataset = []
        fprimes_dataset = []
        energy_dataset = np.array([])
        num_of_atoms = np.array([])
        forces_dataset = []
        index_hashes = []
        self.fp_length = self.fp_length()
        rearange_forces = {}
        n = 0
        for index, atoms_object in enumerate(self.atom_images):
            if self.isamp_hash:
                hash_name = get_amp_hash(atoms_object)
            else:
                hash_name = get_hash(atoms_object, self.Gs)
            index_hashes.append(hash_name)
            image_fingerprint = self.descriptor.fingerprints[hash_name]
            n_atoms = float(len(image_fingerprint))
            num_of_atoms = np.append(num_of_atoms, n_atoms)
            fprange = self.fprange
            atom_order = []
            # fingerprint scaling to [-1,1]
            for i, (atom, afp) in enumerate(image_fingerprint):
                _afp = copy.copy(afp)
                fprange_atom = np.array(fprange[atom])
                for _ in range(np.shape(_afp)[0]):
                    if (fprange_atom[_][1] - fprange_atom[_][0]) > (10.0 ** (-8.0)):
                        _afp[_] = -1 + 2.0 * (
                            (_afp[_] - fprange_atom[_][0])
                            / (fprange_atom[_][1] - fprange_atom[_][0])
                        )
                image_fingerprint[i] = (atom, _afp)
                atom_order.append(atom)
            fingerprint_dataset.append(image_fingerprint)
            image_potential_energy = (
                self.hashed_images[hash_name].get_potential_energy(
                    apply_constraint=False
                )
                / n_atoms
            )
            energy_dataset = np.append(energy_dataset, image_potential_energy)
            if self.forcetraining:
                image_forces = (
                    self.hashed_images[hash_name].get_forces(apply_constraint=False)
                    / n_atoms
                )
                # subtract off delta force contributions
                if self.delta:
                    delta_forces = self.delta_forces[index] / n_atoms
                    image_forces -= delta_forces
                if self.store_primes and os.path.isfile("./stored-primes/" + hash_name):
                    pass
                else:
                    prime_mapping = []
                    for element in self.elements:
                        indices = [i for i, x in enumerate(atom_order) if x == element]
                        prime_mapping += indices
                    new_order = [atom_order[i] for i in prime_mapping]
                    used = set()
                    t = np.array([])
                    for i, x in enumerate(atom_order):
                        for k, l in enumerate(new_order):
                            if (x == l) and (k not in used):
                                used.add(k)
                                t = np.append(t, k)
                                break
                    rearange_forces[index] = t.astype(int)
                    image_primes = self.descriptor.fingerprintprimes[hash_name]
                    # scaling of fingerprint derivatives to be consistent with
                    # fingerprint scaling.
                    _image_primes = copy.copy(image_primes)
                    for _, key in enumerate(list(image_primes.keys())):
                        base_atom = key[3]
                        fprange_atom = np.array(fprange[base_atom])
                        fprange_dif = fprange_atom[:, 1] - fprange_atom[:, 0]
                        fprange_dif[fprange_dif < 10.0 ** (-8.0)] = 2
                        fprime = np.array(image_primes[key])
                        fprime = 2 * fprime / fprange_dif
                        _image_primes[key] = fprime

                    image_prime_values = list(_image_primes.values())
                    image_prime_keys = list(_image_primes.keys())
                    fp_length = len(image_fingerprint[0][1])
                    num_atoms = len(image_fingerprint)
                    if self.specific_atoms:
                        ad_atom_index = get_ad_index(atoms_object)
                    total_atoms_num = len(atoms_object)
                    fingerprintprimes = torch.zeros(
                        fp_length * num_atoms, 3 * total_atoms_num
                    )
                    for idx, fp_key in enumerate(image_prime_keys):
                        image_prime = torch.tensor(image_prime_values[idx])
                        if self.specific_atoms:
                            base_atom = ad_atom_index.index(fp_key[2])
                        else:
                            base_atom = fp_key[2]
                        wrt_atom = fp_key[0]
                        coord = fp_key[4]
                        fingerprintprimes[
                            base_atom * fp_length : base_atom * fp_length + fp_length,
                            wrt_atom * 3 + coord,
                        ] = image_prime
                    # store primes in a sparse matrix format
                    if self.store_primes:
                        sp_matrix = sparse.coo_matrix(fingerprintprimes)
                        sparse.save_npz(
                            open("./stored-primes/" + hash_name, "wb"), sp_matrix
                        )
                    fprimes_dataset.append(fingerprintprimes)
                forces_dataset.append(torch.from_numpy(image_forces))
        if self.delta:
            self.delta_energies /= num_of_atoms
            target_ref_per_atom = energy_dataset[0]
            delta_ref_per_atom = self.delta_energies[0]
            relative_targets = energy_dataset - target_ref_per_atom
            relative_delta = self.delta_energies - delta_ref_per_atom
            energy_dataset = torch.FloatTensor(relative_targets - relative_delta)
            scalings = [target_ref_per_atom, delta_ref_per_atom]
        else:
            energy_dataset = torch.FloatTensor(energy_dataset)
            scalings = [0, 0]
        scale = Transform2(energy_dataset)
        # energy_dataset = scale.norm(energy_dataset)
        if self.forcetraining:
            for idx, force in enumerate(forces_dataset):
                forces_dataset[idx] = scale.norm(force, energy=False)
        scalings.append(scale)

        return (
            fingerprint_dataset,
            energy_dataset,
            num_of_atoms,
            fprimes_dataset,
            forces_dataset,
            index_hashes,
            scalings,
            rearange_forces,
        )


class Transform2():
    def __init__(self, tensor):
        '''
        no normalization
        Parameters
        ----------
        tensor
        '''
        self.mean = 0
        self.std = 1

    def norm(self, tensor, energy=True):
        return (tensor - self.mean) / self.std if energy else tensor/self.std

    def denorm(self, tensor, energy=True):
        return tensor * self.std + self.mean if energy else tensor * self.std


def get_ad_index(image):
    ad_index = []
    for index, atom in enumerate(image):
        if atom.tag == 1:
            ad_index.append(index)
    return ad_index

import torch.nn as nn
class CustomMSELoss_per_image(nn.Module):
    """
    energy loss per image
    """

    def __init__(self, force_coefficient=0):
        super(CustomMSELoss_per_image, self).__init__()
        self.alpha = force_coefficient

    def forward(
        self,
        prediction,
        target):
        energy_pred = prediction[0]
        energy_targets_per_atom = target[0]
        num_atoms = target[1]
        MSE_loss = nn.MSELoss(reduction="mean")
        energy_targets = torch.mul(energy_targets_per_atom, num_atoms)  # per image
        energy_loss = MSE_loss(energy_pred, energy_targets)
        loss = torch.sqrt(energy_loss)
        return loss

def energy_score(net, X, y):
    mse_loss = nn.MSELoss(reduction="mean")
    energy_pred, _ = net.forward(X)
    device = energy_pred.device
    if not hasattr(X, "scalings"):
        X = X.dataset
    scale = X.scalings[-1]
    num_atoms = torch.FloatTensor(np.concatenate(y[1::3])).reshape(-1, 1).to(device)
    dataset_size = len(energy_pred)
    energy_targets_per_atom = torch.tensor(np.concatenate(y[0::3])).to(device).reshape(-1, 1)
    energy_targets_per_image = torch.mul(energy_targets_per_atom, num_atoms)
    energy_loss = mse_loss(energy_pred, energy_targets_per_image)
    energy_rmse = torch.sqrt(energy_loss)
    return energy_rmse


from ase.calculators.calculator import Calculator
from torch.utils.data import DataLoader
from amptorch.utils import hash_images
from amptorch.data_preprocess import TestDataset
class AMPCalculator(Calculator):
    def __init__(self, training_data, model, label, save_fps=False, save_logs=True, specific_atoms=False):
        Calculator.__init__(self)
        self.training_data = training_data
        self.model = model
        self.label = "".join(["results/trained_models/", label, ".pt"])
        self.specific_atoms = specific_atoms
        self.save_fps = save_fps
        self.save_logs = save_logs
        self.testlabel = label
        self.scalings = training_data.scalings
        self.target_ref_per_atom = self.scalings[0]
        self.delta_ref_per_atom = self.scalings[1]
        self.scale = self.scalings[2]
        self.delta = training_data.delta
        self.Gs = training_data.Gs
        self.fprange = training_data.fprange
        self.descriptor = training_data.base_descriptor
        self.cores = training_data.cores
        self.implemented_properties = ["energy", "forces"]
        self.element = training_data.elements
        self.unique_atoms = training_data.elementsall


    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        dataset = TestDataset_seperate(
            images=atoms,
            elements = self.element,
            unique_atoms=self.unique_atoms,
            descriptor=self.training_data.base_descriptor,
            Gs=self.Gs,
            fprange=self.fprange,
            label=self.testlabel,
            cores=self.cores,
            save_fps=self.save_fps,
            specific_atoms=self.specific_atoms
        )
        num_atoms = len(atoms)
        batch_size = len(dataset)
        dataloader = DataLoader(
            dataset, batch_size, collate_fn=dataset.collate_test, shuffle=False
        )
        model = self.model.module
        model.forcetraining = True
        model.load_state_dict(torch.load(self.label))
        model.eval()

        for inputs in dataloader:
            if self.specific_atoms is True:
                unique_atoms = inputs[2]
                num_atoms = inputs[3][0]
            for element in unique_atoms:
                inputs[0][element][0] = inputs[0][element][0].requires_grad_(True)
            energy, forces = model(inputs)
            energy = energy*self.scale.std + self.scale.mean*num_atoms
            forces = self.scale.denorm(forces, energy=False)
        energy = np.concatenate(energy.detach().numpy())
        forces = forces.detach().numpy()

        image_hash = hash_images([atoms])
        if self.delta:
            self.delta_model.neighborlist.calculate_items(image_hash)
            delta_energy, delta_forces, _ = self.delta_model.image_pred(atoms)
            delta_energy = np.squeeze(delta_energy)
            energy += delta_energy + num_atoms*(self.target_ref_per_atom - self.delta_ref_per_atom)
            forces += delta_forces

        self.results["energy"] = float(energy)
        self.results["forces"] = forces

class AtomsDataset_ZEO_AD(AtomsDataset):
    """
    deleted the normalization
    get loss per image
    """

    def __init__(
            self,
            images,
            descriptor,
            Gs,
            forcetraining,
            label,
            cores,
            delta_data=None,
            store_primes=False,
            specific_atoms=False
    ):
        self.images = images
        self.base_descriptor = descriptor
        self.descriptor = descriptor
        self.Gs = Gs
        self.atom_images = self.images
        self.forcetraining = forcetraining
        self.store_primes = store_primes
        self.cores = cores
        self.delta = False
        self.specific_atoms = specific_atoms
        if delta_data is not None:
            self.delta_data = delta_data
            self.delta_energies = np.array(delta_data[0])
            self.delta_forces = delta_data[1]
            self.num_atoms = np.array(delta_data[2])
            self.delta = True
        if self.store_primes:
            if not os.path.isdir("./stored-primes/"):
                os.mkdir("stored-primes")
        if isinstance(images, str):
            extension = os.path.splitext(images)[1]
            if extension != (".traj" or ".db"):
                self.atom_images = ase.io.read(images, ":")
        self.elements = self.unique()
        # TODO Print log - control verbose
        print("Calculating fingerprints...")
        G2_etas = Gs["G2_etas"]
        G2_rs_s = Gs["G2_rs_s"]
        G4_etas = Gs["G4_etas"]
        G4_zetas = Gs["G4_zetas"]
        G4_gammas = Gs["G4_gammas"]
        cutoff = Gs["cutoff"]
        if descriptor == SNN_Gaussian:
            # create simple_nn fingerprints
            self.hashed_images = hash_images(self.atom_images, Gs=Gs)
            make_amp_descriptors_simple_nn_seperate(
                self.atom_images, Gs, self.elements, cores=cores, label=label, specific_atoms=self.specific_atoms
            )
            self.isamp_hash = False
        else:
            self.hashed_images = amp_hash(self.atom_images)
            self.isamp_hash = True
        G = make_symmetry_functions(elements=self.elements, type="G2", etas=G2_etas)
        G += make_symmetry_functions(
            elements=self.elements,
            type="G4",
            etas=G4_etas,
            zetas=G4_zetas,
            gammas=G4_gammas,
        )
        for g in list(G):
            g["Rs"] = G2_rs_s
        self.descriptor = self.descriptor(Gs=G, cutoff=cutoff)
        self.descriptor.calculate_fingerprints(
            self.hashed_images, calculate_derivatives=forcetraining
        )
        print("Fingerprints Calculated!")
        self.fprange = calculate_fingerprints_range(self.descriptor, self.hashed_images)
        # perform preprocessing
        self.fingerprint_dataset, self.energy_dataset, self.num_of_atoms, self.sparse_fprimes, self.forces_dataset, self.index_hashes, self.scalings, self.rearange_forces = (
            self.preprocess_data()
        )
        self.elementsall = self.unique2()

    def unique2(self):
        """Returns the unique elements contained in the training dataset"""
        elements = []
        for image in self.images:
            for atom in image:
                if atom.tag == 1 and atom.symbol not in elements:
                    elements.append(atom.symbol)
                elif atom.tag == 2:
                    zsym = 'z' + atom.symbol
                    if zsym not in elements:
                        elements.append(zsym)
        return elements

    def preprocess_data(self):
        # TODO cleanup/optimize
        fingerprint_dataset = []
        fprimes_dataset = []
        energy_dataset = np.array([])
        num_of_atoms = np.array([])
        forces_dataset = []
        index_hashes = []
        self.fp_length = self.fp_length()
        rearange_forces = {}
        n = 0
        for index, atoms_object in enumerate(self.atom_images):
            if self.isamp_hash:
                hash_name = get_amp_hash(atoms_object)
            else:
                hash_name = get_hash(atoms_object, self.Gs)
            index_hashes.append(hash_name)
            image_fingerprint = self.descriptor.fingerprints[hash_name]
            n_atoms = float(len(image_fingerprint))
            num_of_atoms = np.append(num_of_atoms, n_atoms)
            fprange = self.fprange
            atom_order = []
            # fingerprint scaling to [-1,1]
            for i, (atom, afp) in enumerate(image_fingerprint):
                _afp = copy.copy(afp)
                fprange_atom = np.array(fprange[atom])
                for _ in range(np.shape(_afp)[0]):
                    if (fprange_atom[_][1] - fprange_atom[_][0]) > (10.0 ** (-8.0)):
                        _afp[_] = -1 + 2.0 * (
                            (_afp[_] - fprange_atom[_][0])
                            / (fprange_atom[_][1] - fprange_atom[_][0])
                        )
                image_fingerprint[i] = (atom, _afp)
                atom_order.append(atom)
            fingerprint_dataset.append(image_fingerprint)
            image_potential_energy = (
                self.hashed_images[hash_name].get_potential_energy(
                    apply_constraint=False
                )
                / n_atoms
            )
            energy_dataset = np.append(energy_dataset, image_potential_energy)
            if self.forcetraining:
                image_forces = (
                    self.hashed_images[hash_name].get_forces(apply_constraint=False)
                    / n_atoms
                )
                # subtract off delta force contributions
                if self.delta:
                    delta_forces = self.delta_forces[index] / n_atoms
                    image_forces -= delta_forces
                if self.store_primes and os.path.isfile("./stored-primes/" + hash_name):
                    pass
                else:
                    prime_mapping = []
                    for element in self.elements:
                        indices = [i for i, x in enumerate(atom_order) if x == element]
                        prime_mapping += indices
                    new_order = [atom_order[i] for i in prime_mapping]
                    used = set()
                    t = np.array([])
                    for i, x in enumerate(atom_order):
                        for k, l in enumerate(new_order):
                            if (x == l) and (k not in used):
                                used.add(k)
                                t = np.append(t, k)
                                break
                    rearange_forces[index] = t.astype(int)
                    image_primes = self.descriptor.fingerprintprimes[hash_name]
                    # scaling of fingerprint derivatives to be consistent with
                    # fingerprint scaling.
                    _image_primes = copy.copy(image_primes)
                    for _, key in enumerate(list(image_primes.keys())):
                        base_atom = key[3]
                        fprange_atom = np.array(fprange[base_atom])
                        fprange_dif = fprange_atom[:, 1] - fprange_atom[:, 0]
                        fprange_dif[fprange_dif < 10.0 ** (-8.0)] = 2
                        fprime = np.array(image_primes[key])
                        fprime = 2 * fprime / fprange_dif
                        _image_primes[key] = fprime

                    image_prime_values = list(_image_primes.values())
                    image_prime_keys = list(_image_primes.keys())
                    fp_length = len(image_fingerprint[0][1])
                    num_atoms = len(image_fingerprint)
                    if self.specific_atoms:
                        ad_atom_index = get_ad_index(atoms_object)
                    total_atoms_num = len(atoms_object)
                    fingerprintprimes = torch.zeros(
                        fp_length * num_atoms, 3 * total_atoms_num
                    )
                    for idx, fp_key in enumerate(image_prime_keys):
                        image_prime = torch.tensor(image_prime_values[idx])
                        if self.specific_atoms:
                            base_atom = ad_atom_index.index(fp_key[2])
                        else:
                            base_atom = fp_key[2]
                        wrt_atom = fp_key[0]
                        coord = fp_key[4]
                        fingerprintprimes[
                            base_atom * fp_length : base_atom * fp_length + fp_length,
                            wrt_atom * 3 + coord,
                        ] = image_prime
                    # store primes in a sparse matrix format
                    if self.store_primes:
                        sp_matrix = sparse.coo_matrix(fingerprintprimes)
                        sparse.save_npz(
                            open("./stored-primes/" + hash_name, "wb"), sp_matrix
                        )
                    fprimes_dataset.append(fingerprintprimes)
                forces_dataset.append(torch.from_numpy(image_forces))
        if self.delta:
            self.delta_energies /= num_of_atoms
            target_ref_per_atom = energy_dataset[0]
            delta_ref_per_atom = self.delta_energies[0]
            relative_targets = energy_dataset - target_ref_per_atom
            relative_delta = self.delta_energies - delta_ref_per_atom
            energy_dataset = torch.FloatTensor(relative_targets - relative_delta)
            scalings = [target_ref_per_atom, delta_ref_per_atom]
        else:
            energy_dataset = torch.FloatTensor(energy_dataset)
            scalings = [0, 0]
        scale = Transform2(energy_dataset)
        # energy_dataset = scale.norm(energy_dataset)
        if self.forcetraining:
            for idx, force in enumerate(forces_dataset):
                forces_dataset[idx] = scale.norm(force, energy=False)
        scalings.append(scale)

        return (
            fingerprint_dataset,
            energy_dataset,
            num_of_atoms,
            fprimes_dataset,
            forces_dataset,
            index_hashes,
            scalings,
            rearange_forces,
        )


class TestDataset_seperate(TestDataset):
    """
    PyTorch inherited abstract class that specifies how testing data is to be preprocessed and
    passed along to the DataLoader.

    Parameters:
    -----------
    images: list, file, database
        Training data to be utilized for the regression model.

    descriptor: object
        Scheme to be utilized for computing fingerprints

    fprange: Dict
        Fingerprint ranges of the training dataset to be used to scale the test
        dataset fingerprints in the same manner.

    """

    def __init__(self, images, elements, unique_atoms, descriptor, Gs, fprange, label="example", cores=1,
                 save_fps=False, specific_atoms=False):
        super().__init__(images, unique_atoms, descriptor, Gs, fprange, label, cores, save_fps, specific_atoms)
        self.images = images
        if type(images) is not list:
            self.images = [images]
        self.descriptor = descriptor
        self.atom_images = self.images
        if isinstance(images, str):
            extension = os.path.splitext(images)[1]
            if extension != (".traj" or ".db"):
                self.atom_images = ase.io.read(images, ":")
        self.fprange = fprange
        self.save_fps = save_fps
        self.elements = elements
        self.unique_atoms = unique_atoms
        self.hashed_images = amp_hash(self.atom_images)
        self.specific_atoms = specific_atoms
        G2_etas = Gs["G2_etas"]
        G2_rs_s = Gs["G2_rs_s"]
        G4_etas = Gs["G4_etas"]
        G4_zetas = Gs["G4_zetas"]
        G4_gammas = Gs["G4_gammas"]
        cutoff = Gs["cutoff"]
        if descriptor == SNN_Gaussian:
            # create simple_nn fingerprints
            self.hashed_images = hash_images(self.atom_images, Gs=Gs)
            self.fps, self.fp_primes = make_amp_descriptors_simple_nn_seperate(
                self.atom_images,
                Gs,
                self.elements,
                cores=cores,
                label=label,
                save=self.save_fps,
                specific_atoms=self.specific_atoms
            )
            self.isamp_hash = False
        else:
            self.hashed_images = amp_hash(self.atom_images)
            self.isamp_hash = True
        if self.save_fps:
            G = make_symmetry_functions(elements=self.elements, type="G2", etas=G2_etas)
            G += make_symmetry_functions(
                elements=self.elements,
                type="G4",
                etas=G4_etas,
                zetas=G4_zetas,
                gammas=G4_gammas,
            )
            for g in list(G):
                g["Rs"] = G2_rs_s
            self.descriptor = self.descriptor(Gs=G, cutoff=cutoff)
            self.descriptor.calculate_fingerprints(
                self.hashed_images, calculate_derivatives=True
            )


def convert_simple_nn_fps(traj, Gs, specific_atoms, cores, label, save, delete_old=True):
    from multiprocessing import Pool

    # make the directories
    if not os.path.isdir("./amp-data-fingerprints.ampdb"):
        os.mkdir("./amp-data-fingerprints.ampdb")
    if not os.path.isdir("./amp-data-fingerprints.ampdb/loose"):
        os.mkdir("./amp-data-fingerprints.ampdb/loose")
    if not os.path.isdir("./amp-data-fingerprint-primes.ampdb"):
        os.mkdir("./amp-data-fingerprint-primes.ampdb")
    if not os.path.isdir("./amp-data-fingerprint-primes.ampdb/loose"):
        os.mkdir("amp-data-fingerprint-primes.ampdb/loose")
    # perform the reorganization
    l_trajs = list(enumerate(traj))
    fp_dir = "data"+label
    '''
    if len(traj) > 1:
        with Pool(cores) as p:
            l_trajs = [image + (Gs, specific_atoms, label, fp_dir) for image in l_trajs]
            p.map(reorganize, l_trajs)
    else:
        image = (0, traj[0], Gs, specific_atoms, label, fp_dir)
        fps, fp_primes = reorganize(image, save=save)
    '''
    if len(traj) > 1:
        l_trajs = [image + (Gs, specific_atoms, label, fp_dir) for image in l_trajs]
        for i, inp in enumerate(l_trajs):
            reorganize_seperate(inp)
    else:
        image = (0, traj[0], Gs, specific_atoms, label, fp_dir)
        fps, fp_primes = reorganize_seperate(image, save=save)
    if delete_old:
        os.rmdir("./data"+label)
    if save:
        return None, None
    else:
        return fps, fp_primes


def reorganize_seperate(inp, delete_old=True, save=True):
    i, image, Gs, specific_atoms, label, fp_dir = inp
    pic = pickle.load(open(fp_dir+"/data{}.pickle".format(i + 1), "rb"))
    im_hash = get_hash(image, Gs)
    x_list = reorganize_simple_nn_fp(image, specific_atoms, pic["x"])
    #x_der_dict = reorganize_simple_nn_derivative(image, specific_atoms, pic["dx"])
    x_der_dict = {}
    if save:
        pickle.dump(x_list, open("./amp-data-fingerprints.ampdb/loose/" + im_hash, "wb"))
        pickle.dump(
            x_der_dict, open("./amp-data-fingerprint-primes.ampdb/loose/" + im_hash, "wb")
        )
    if delete_old:  # in case disk space is an issue
        if os.path.exists(os.path.join("./data" + label,'simple_nn_log')):
            os.remove(os.path.join("./data" + label,'simple_nn_log'))
        os.remove(fp_dir+"/data{}.pickle".format(i + 1))
    return x_list, x_der_dict

def reorganize_simple_nn_fp(image, specific_atoms, x_dict):
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
    sym_dict = OrderedDict()
    if specific_atoms is True:
        syms = image.get_chemical_symbols()
        for sym in syms:
            sym_dict[sym] = []
        for i, sym in enumerate(syms):
            if image[i].tag == 1:
                sym_dict[sym].append(i)
        for i, sym in enumerate(syms):
            if image[i].tag == 1:
                simple_nn_index = sym_dict[sym].index(i)
                fp = x_dict[sym][simple_nn_index]
                fp_l.append((sym, list(fp)))
        # slab atoms tag == 2
        for sym in syms:
            key_slab = 'z' + sym
            sym_dict[key_slab] = []
        for i, sym in enumerate(syms):
            if image[i].tag == 2:
                key_slab = 'z' + sym
                sym_dict[key_slab].append(i)
        for i, sym in enumerate(syms):
            if image[i].tag == 2:
                key_slab = 'z' + sym
                simple_nn_index = sym_dict[key_slab].index(i)
                fp = x_dict[sym][simple_nn_index]
                fp_l.append((key_slab, list(fp)))
    else:
        syms = image.get_chemical_symbols()
        for sym in syms:
            sym_dict[sym] = []
        for i, sym in enumerate(syms):
            sym_dict[sym].append(i)
        for i, sym in enumerate(syms):
            simple_nn_index = sym_dict[sym].index(i)
            fp = x_dict[sym][simple_nn_index]
            fp_l.append((sym, list(fp)))
    return fp_l

def make_amp_descriptors_simple_nn_seperate(atoms, Gs, elements, cores, label, save=True, specific_atoms=False):
    """
    uses simple_nn to make descriptors in the amp format.
    Only creates the same symmetry functions for each element
    for now.
    """
    traj, calculated = make_simple_nn_fps(atoms, Gs, specific_atoms, elements=elements,
            label=label, clean_up_directory=True)
    if calculated:
        fps, fp_primes = convert_simple_nn_fps(traj, Gs, specific_atoms, cores, label, save, delete_old=True)
        return fps, fp_primes
    if save is False and calculated is False:
        fps, fp_primes = stored_fps(atoms, Gs)
        return fps, fp_primes
    else: return None, None
