import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

class simulation:
    """
    Class for storing information relevant to a single umbrella+metadynamics simulation.
    """
    def __init__(self, COLVAR_file, target_dist, beta, k, f=0.0):
        self.CV_data = np.genfromtxt(COLVAR_file)
        self.target_dist = target_dist
        self.beta = beta
        self.k = k
        self.f = f

    def compute_Puh(self, x_col=1, y_col=2, rbias_col=5, xbins=100, ybins=100, xmin=1.1, xmax=4.2, ymin=0.0, ymax=2.75):
        """
        Computed probability distributions reweighted with respect to time-dependent metadynamics bias using eqn 13 in Nair paper.
        Reweighted probability is computed for a single metadynamics simulation and attached to a simulation object as an instance variable.
        """
        histogram, xedges, yedges = np.histogram2d(x=self.CV_data[:, x_col], y=self.CV_data[:, y_col], bins=(xbins,ybins), weights=np.exp(self.CV_data[:, rbias_col]*self.beta), density=True, range=[(xmin,xmax),(ymin,ymax)])
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        self.prob_uh = histogram.T
        self.xedges, self.yedges = xedges, yedges
        self.xcenters, self.ycenters = xcenters, ycenters
        self.n_samples = self.CV_data.shape[0]


class WSMD_reweighting:
    """
    Class for computing reweighting of well-sliced metadynamics simulations
    """
    def __init__(self, target_dists, COLVAR_paths, beta, k):
        """
        Load all the CV data into a list of simulation objects
        """
        self.beta = beta
        self.k = k
        simulation_data_list = []
        for target_dist, COLVAR_path in zip(target_dists, COLVAR_paths):
            simulation_data = simulation(COLVAR_file=COLVAR_path, target_dist=target_dist, beta=self.beta, k=self.k)
            simulation_data_list.append(simulation_data)
        self.simulation_data_list = simulation_data_list
        
    def compute_probs(self):
        """
        Reweights the time-dependent metadynamics bias for each simulation according to eqn 13 in Nair paper using c(t)'s computed by Plumed.
        """
        for simulation in self.simulation_data_list:
            print ('Computing prob_uh for sim: %s' % simulation)
            simulation.compute_Puh()

        self.xedges = self.simulation_data_list[0].xedges
        self.yedges = self.simulation_data_list[0].yedges
        self.xcenters = self.simulation_data_list[0].xcenters
        self.ycenters = self.simulation_data_list[0].ycenters
        
    def compute_top_eqn(self):
        """
        Computes the top sum in eqn 14 in Nair paper.
        Stores the output in the WSMD object.
        """
        for index, simulation in enumerate(self.simulation_data_list):
            if index == 0:
                top_eqn = np.zeros_like(simulation.prob_uh)
                top_eqn += simulation.n_samples*simulation.prob_uh
            else:
                top_eqn += simulation.n_samples*simulation.prob_uh
        self.top_eqn = top_eqn

    def umbrella_potential(self, s, target_dist, k):
        """
        Computes umbrella potential bias
        """
        return (0.5*k*(s - target_dist)*(s - target_dist))

    def compute_bot_eqn(self):
        """
        Computes the bottom sum in eqn 14 in Nair paper.
        """
        bot_eqn = 0
        for index, simulation in enumerate(self.simulation_data_list):
            bot_eqn += simulation.n_samples*np.exp(simulation.beta*simulation.f)*np.exp(-1*simulation.beta*self.umbrella_potential(simulation.xcenters, simulation.target_dist, simulation.k))
        return bot_eqn

    def compute_trial_prob(self, top_eqn, bot_eqn):
        """
        Combines the outputs of "compute_top_eqn" and "compute_bot_eqn" to compute the full rhs of eqn 14 in Nair paper.
        Returns a trial 2D probability distribution for WHAM equation SCF.
        """
        bot_eqn_invert = np.reciprocal(bot_eqn)
        bot_matrix = np.array([bot_eqn_invert] * top_eqn.shape[0])
        prob = np.multiply(top_eqn,bot_matrix)
        return prob

    def compute_trial_fs(self, prob):
        """
        Computes eqn 15 in Nair paper.
        Returns a list of new f constants for WHAM SCF.
        """
        for index, simulation in enumerate(self.simulation_data_list):
            Z = np.multiply(np.tile(np.exp(-1*simulation.beta*self.umbrella_potential(simulation.xcenters, simulation.target_dist, simulation.k)), (prob.shape[0],1)), prob)
            int_Z = simps(simps(Z, simulation.xcenters), simulation.ycenters)
            f = -1*(1/simulation.beta)*np.log(int_Z)
            simulation.f = f
        return [simulation.f for simulation in self.simulation_data_list]

    def WHAM_calc(self, max_iters=10000, threshold=1e-4):
        """
        Self-consistently solves the WHAM equations (eqns 14 and 15 in Nair paper) in order to switch together sampling along the umbrella coordinate.
        Returns the 2D free energy surface on a grid.
        """
        self.compute_probs()
        self.compute_top_eqn()
        
        for step in range(max_iters):
            bot_eqn = self.compute_bot_eqn()
            trial_prob = self.compute_trial_prob(self.top_eqn, bot_eqn)
            initial_fs = np.array([simulation.f for simulation in self.simulation_data_list])
            trial_fs = self.compute_trial_fs(trial_prob)
            diff = initial_fs - trial_fs
            average_diff = np.mean(np.abs(diff))
            print ('step %s, diff %s' % (step, average_diff))
            if average_diff < threshold:
                print ('convergence threshold met in %s steps, computing FES...' % step)
                self.final_prob = trial_prob
                break
            if step == max_iters-1:
                print ('max steps hit, computing FES...')
                self.final_prob = trial_prob

        FES = -1*(1/self.beta)*np.log(self.final_prob)
        FES[np.isneginf(FES)] = 0
        FES -= np.min(FES)
        self.FES = FES

    def plot_FES(self, filename='2d_FES_WSMD_ACN.png', cmap='jet', vmin=0.0, vmax=80.0, xlabel='C-C Bond Distance ($\AA$)', ylabel='MODB-COM/BF4 CN (4.0 $\AA$)', title='WSMD in ACN/TEMA/BF4', xmin=1.0, xmax=4.5, ymin=0.0, ymax=2.5, colorbar_label='Free Energy (kJ/mol)', dpi=300):
        """
        Plots the 2D free energy surface computed via WSMD.
        """
        fig, ax = plt.subplots()
        c = ax.pcolormesh(self.xedges, self.yedges, self.FES, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(xmin,xmax), ylim=(ymin,ymax))
        fig.colorbar(c, ax=ax, label=colorbar_label)
        plt.savefig(filename, dpi=dpi)
        plt.close()
        
def main():
    """
    Computes the FES with respect to two coordinates using well-sliced metadynamics.
    Need to input value of beta (in appropriate energy units) and k (in energy/angstrom^2). Also need umbrella sampling coordinates and data sampled using umbrella+metadynamics.
    """
    beta = 1/2.479 # KbT = 2.479 kJ/mol
    k = 1200.0 # kJ/mol/angstrom^2
    COLVAR_filename = 'COLVAR'    

    home = os.getcwd()
    dirs = [f for f in os.listdir(home) if f.startswith('dist_')] # List of directories where each COLVAR file is stored
    dirs.sort()
    
    target_dists = []
    COLVAR_paths = []
    for d in dirs:
        target_dist = float(d.split('_')[1])*10
        COLVAR_path = os.path.join(home, d, COLVAR_filename)
        target_dists.append(target_dist)
        COLVAR_paths.append(COLVAR_path)
    
    WSMD = WSMD_reweighting(target_dists, COLVAR_paths, beta=beta, k=k)
    WSMD.WHAM_calc()
    WSMD.plot_FES() 

if __name__ == "__main__":
    main()



