from sigopt import Connection
from sklearn.metrics import mean_squared_error
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ase.visualize import view
import ase.io
from amp import Amp
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
import operator
import amptorch
import copy
import matplotlib
from skorch import NeuralNetRegressor
from skorch.dataset import CVSplit
from skorch.callbacks import Checkpoint, EpochScoring
from skorch.callbacks.lr_scheduler import LRScheduler
import skorch.callbacks.base
from amptorch.gaussian import SNN_Gaussian
from amptorch.model import BPNN, CustomMSELoss
from amptorch.data_preprocess import AtomsDataset, factorize_data, collate_amp, TestDataset
from amptorch.skorch_model import AMP
from amptorch.skorch_model.utils import target_extractor, energy_score, forces_score
from amptorch.analysis import parity_plot
from torch.utils.data import DataLoader
from torch.nn import init
from skorch.utils import to_numpy
import matplotlib.pyplot as plt
import os
conn = Connection(client_token="HKIERIUWCZZNRSGUKWXLXHXHIISCFYUICPJDBPMOLDQZVRZH")


def Select_image(images):
    train_image = []
    test_image = []
    for index,image in enumerate(images[:]):
        if index-index//3*3 == 0:
            test_image.append(image)
        else:
            train_image.append(image)
    print(len(train_image), len(test_image))
    return train_image, test_image

experiment = conn.experiments().create(
    name="adsorption_zeolite-2",
    metrics=[
        dict(name="energy_rmse_test", objective="minimize"),
        dict(name="energy_rmse_train", objective="minimize"),
    ],
    parameters=[
        dict(name="hidden_layers", bounds=dict(min=2, max=30), type="int"),
        dict(name="num_nodes", bounds=dict(min=2, max=100), type="int"),
        dict(name="learning_rate", bounds=dict(min=1e-4, max=0.1), type="double"),
        dict(name="epochs", bounds=dict(min=5, max=1000), type="int"),
    ],
    observation_budget=50,
    project="amptorch",
)
print("Create experiment:" + experiment.id)



def run_model(assignments, numberi=0):
    class train_end_load_best_valid_loss(skorch.callbacks.base.Callback):
        def on_train_end(self, net, X, y):
            net.load_params('valid_best_params.pt')
    out_file = open('log-sigopt','a')
    # LR_schedule = LRScheduler("CosineAnnealingLR", T_max=assignments["T_max"])
    cp = Checkpoint(monitor='valid_loss_best', fn_prefix='valid_best_')
    load_best_valid_loss = train_end_load_best_valid_loss()
    images = ase.io.read('../traj_taged_adsorptionenergy.traj', index=':')
    Gs = {}
    Gs["G2_etas"] = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
    Gs["G2_rs_s"] = [0] * 4
    Gs["G4_etas"] = [0.005]
    Gs["G4_zetas"] = [1.0]
    Gs["G4_gammas"] = [+1.0, -1]
    Gs["cutoff"] = 6.5
    forcetraining = False
    label = "sigopt-zeolite"
    train_images, test_images = Select_image(images)

    DFT_energies_test = [image.get_potential_energy() for image in test_images]
    DFT_energies_train = [image.get_potential_energy() for image in train_images]
    training_data = AtomsDataset(train_images, SNN_Gaussian, Gs, forcetraining=forcetraining,
            label=label, cores=1, delta_data=None, specific_atoms=True)

    unique_atoms = training_data.elements
    fp_length = training_data.fp_length
    device = "cpu"
    torch.set_num_threads(1)
    optimizer = optim.LBFGS
    batch_size = len(training_data)
    net = NeuralNetRegressor(
        module=BPNN(
            unique_atoms,
            [fp_length, assignments["hidden_layers"], assignments["num_nodes"]],
            device,
            forcetraining=forcetraining,
        ),
        criterion=CustomMSELoss,
        criterion__force_coefficient=0,
        optimizer=optimizer,
        #optimizer=torch.optim.LBFGS,
        lr=assignments["learning_rate"],
        #lr=1e-1,
        batch_size=batch_size,
        max_epochs=assignments["epochs"],
        iterator_train__collate_fn=collate_amp,
        iterator_train__shuffle=False,
        iterator_valid__collate_fn=collate_amp,
        iterator_valid__shuffle=False,
        device=device,
        train_split=CVSplit(cv=0.2),
        callbacks=[
            EpochScoring(
                energy_score,
                on_train=True,
                use_caching=True,
                target_extractor=target_extractor,
            ),
             cp,
             load_best_valid_loss,
            # LR_schedule
        ],
    )
    calc = AMP(training_data, net, label,specific_atoms=True)
    calc.train(overwrite=True)
    for image in test_images:
        image.set_calculator(calc)
    for image in train_images:
        image.set_calculator(calc)
    pred_energies_test = [image.get_potential_energy() for image in test_images]
    pred_energies_train = [image.get_potential_energy() for image in train_images]
    for e_t in pred_energies_test:
        if np.isnan(e_t):
            e_t = 1e10
    for e_tr in pred_energies_train:
        if np.isnan(e_tr):
            e_tr =1e10
    energy_rmse_test = mean_squared_error(pred_energies_test, DFT_energies_test)
    energy_rmse_train = mean_squared_error(pred_energies_train, DFT_energies_train)
    print('***************', file=out_file)
    print('No. ', numberi, file=out_file)
    print('****************************RMSE_TEST:', energy_rmse_test,file=out_file)
    print('****************************RMSE_Train:', energy_rmse_train,file=out_file)
    print('Energies:::::::', file=out_file)
    print('train data',file=out_file)
    for i in range(len(train_images)):
        print('(',DFT_energies_train[i],';',pred_energies_train[i],') ', end='',file=out_file)
    print('****',file=out_file)
    print('test data', file=out_file)
    for i in range(len(test_images)):
        print('(',DFT_energies_test[i],';',pred_energies_test[i],') ', end='',file=out_file)
    print('****',file=out_file)
    plt.cla()
    fig = plt.figure('ML vs DFT')
    ax = fig.add_subplot(111)
    ax.scatter(DFT_energies_train, pred_energies_train, c='b')
    ax.scatter(DFT_energies_test, pred_energies_test, c='r')
    ax.set_xlabel('E_DFT')
    ax.set_ylabel('E_ML')
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    figure_name = './figures/DFTvsML'+ str(numberi)+ '.png'
    fig.savefig(figure_name, bbox_inches='tight')
    return energy_rmse_test, energy_rmse_train


for _ in range(experiment.observation_budget):
    suggestion = conn.experiments(experiment.id).suggestions().create()
    assignments = suggestion.assignments
    print(assignments)
    energy_rmse_test, energy_rmse_train = run_model(assignments, _)
    values = [
        {"name": "energy_rmse_test", "value": energy_rmse_test},
        {"name": "energy_rmse_train", "value": energy_rmse_train},
    ]
    if np.isnan(energy_rmse_test):
        energy_rmse_test = 1e10
    if np.isnan(energy_rmse_train):
        energy_rmse_train = 1e10
    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id, values=values
    )
    assignments = (
        conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments
    )
