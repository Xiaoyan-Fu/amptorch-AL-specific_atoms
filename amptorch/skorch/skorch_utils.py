import ase
from ase import units
from ase.md import Langevin
import copy
from ase.calculators.emt import EMT
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def md_run(calc, starting_image, temp, count, label):
    traj = ase.io.Trajectory(label+".traj", "w")
    slab = starting_image.copy()
    slab.set_calculator(calc)
    slab.get_forces()
    traj.write(slab)
    dyn = Langevin(slab, 1.0 * units.fs, temp * units.kB, 0.002)
    for step in range(count-1):
        dyn.run(20)
        traj.write(slab)

def calculate_energies(emt_images, ml_images):
    energies_emt = [image.get_potential_energy() for image in emt_images]
    ml_energies_apparent = [image.get_potential_energy() for image in ml_images]
    ml_energies_actual = []
    for image in ml_images:
        image = copy.copy(image)
        image.set_calculator(EMT())
        ml_energies_actual.append(image.get_potential_energy())
    return energies_emt, ml_energies_apparent, ml_energies_actual

def calculate_forces(emt_images, ml_images, type="max"):
    if type == "max":
        forces_emt = [
            np.log10((np.array(np.amax(np.abs(image.get_forces())))))
            for image in emt_images
        ]
        ml_forces_apparent = [
            np.log10((np.array(np.amax(np.abs(image.get_forces())))))
            for image in ml_images
        ]
        ml_forces_actual = []
        for image in ml_images:
            image = copy.copy(image)
            image.set_calculator(EMT())
            ml_forces_actual.append(
                np.log10((np.array(np.amax(np.abs(image.get_forces())))))
            )
    else:
        forces_emt = [image.get_forces() for image in emt_images]
        ml_forces_apparent = [image.get_forces() for image in ml_images]
        ml_forces_actual = []
        for image in ml_images:
            image = copy.copy(image)
            image.set_calculator(EMT())
            ml_forces_actual.append(image.get_forces())
    return forces_emt, ml_forces_apparent, ml_forces_actual

def time_plots(target, preds, sampled_points, label, property="energy", filename=None):
    fig, ax = plt.subplots(figsize=(14.15, 10))
    time = np.linspace(0, 20 * len(target), len(target))
    if sampled_points:
        sampled_time = [time[i] for i in sampled_points]
        samples = [preds[0][i] for i in sampled_points]
        plt.plot(sampled_time, samples, "o", label="sampled points")
    plt.plot(time, target, color="k", label="target")
    for idx, data in enumerate(preds):
        plt.plot(time, data, label=label[idx])
    plt.xlabel("time, (ps)", fontsize=25)
    if property == "energy":
        plt.ylabel("energy, eV", fontsize=25)
    else:
        plt.ylabel("logmax|F|", fontsize=25)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

def kde_plots(emt, forces, labels, filename=None):
    fig, ax = plt.subplots(figsize=(14.15, 10))
    sns.distplot(
        emt,
        hist=False,
        kde=True,
        label="target",
        color="k",
        hist_kws={"edgecolor": "black", "alpha": 0.1},
        kde_kws={"linewidth": 2},
        ax=ax,
    )
    for i, data in enumerate(forces):
        sns.distplot(
            data,
            hist=False,
            label=labels[i],
            kde=True,
            hist_kws={"edgecolor": "black", "alpha": 0.1},
            kde_kws={"linewidth": 2},
        )
    plt.legend(loc="best", fontsize=28)
    plt.xlabel("log max|F|", fontsize=32)
    plt.ylabel("Density Function", fontsize=32)
    plt.title("Force Distributions", fontsize=35)
    plt.xlim(left=-5, right=4)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()

