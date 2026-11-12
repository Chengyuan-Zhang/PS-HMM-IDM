import argparse
import numpy as np


class Config:
    def __init__(self):
        # HighD preprocessing
        self.frame_rate_orignial = 25
        self.frame_rate = 5
        self.dt = 1 / self.frame_rate
        self.min_traj_lenth = 50 * self.frame_rate_orignial

        # HMM
        self.K = 5  # 2/5/10/25/50/75/100
        self.KS = 5  # scenario number
        self.dirichlet_alpha = 1000  # 1/500/1000/2000/3000/4000/5000/10000
        self.emission_func = 'MvNormal'  # 'IDM-MvNormal'  # 'IDM' or 'Normal' or 'MvNormal' or 'IDM-MvNormal'
        self.model = 'HMM'  # 'FHMM', 'HMM' or 'GMM'
        self.sampler = 'Slice'  # 'Elliptical'  or 'MH' or 'Slice'

        # IDM
        self.delta = 4
        self.epsilon = 1e-8
        self.mean_params = np.array([33, 2, 1.6, 1.5, 1.67])  # np.exp(np.array([2., 0.69, 0.47, .4, .51]))  #
        self.cov_params = np.diag([0.1, 0.1, 0.1, 0.1, 0.1]) ** 2 + np.eye(5) * self.epsilon

        # Slice sampling
        self.bounds = [(5, 200), (1e-3, 50), (1e-3, 50), (1e-4, 50), (1e-4, 50)]

        # MH sampling
        self.n_chains = 1
        self.n_sample = 1
        self.n_burn_in = 0
        self.proposal_sigma = 0.1  # in logarithm scale, 0.05 in original scale

        # MCMC
        self.m_chains = 1
        self.m_sample = 2000
        self.m_burn_in = 6000
        self.n_jobs = 8
        self.w = 0.1  # in logarithm scale; 0.02 in original scale

        # Priors
        # dirichlet_alpha = 0.01
        self.dim_mixture_part = 3
        self.gamma_alpha = 100  # TODO: 10 or 1
        self.gamma_beta = 1
        self.kappa = 0.01  # need to be tuned
        self.nu_x = self.dim_mixture_part + 2
        self.mu0 = np.zeros(3)
        self.Psi0 = np.eye(3) * 1e-2

        self.nu = 7  # dim(IDM_params) + 2
        self.Psi = np.diag([.1, .1, .1, .1, .1]) + np.eye(5) * 1e-12

        # IOHMM
        self.input_feature = 'multi'  # 'multi' or 'a'

        # Simulation
        self.car_interactive_pair_list = [14, 35, 23, 25, 36, 38, 60, 90, 228, 232]
        self.truck_interactive_pair_list = [3, 4, 18, 52, 81, 144, 153, 162, 5, 241]
        self.pair_list_all = self.car_interactive_pair_list + self.truck_interactive_pair_list

        # Parse command-line arguments to override defaults
        self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser()

        # Add arguments for hyperparameters to test
        parser.add_argument('--K', type=int, default=self.K, choices=[2, 5, 10, 25])
        parser.add_argument('--KS', type=int, default=self.KS, choices=[2, 5, 10, 25])
        parser.add_argument('--dirichlet_alpha', type=int, default=self.dirichlet_alpha,
                            choices=[1, 500, 1000, 2000, 3000, 4000, 5000, 10000])
        parser.add_argument('--emission_func', type=str, default=self.emission_func,
                            choices=['IDM', 'Normal', 'MvNormal', 'IDM-MvNormal'])
        parser.add_argument('--model', type=str, default=self.model, choices=['FHMM', 'GMM'])
        parser.add_argument('--sampler', type=str, default=self.sampler, choices=['Elliptical', 'MH', 'Slice'])

        # args = parser.parse_args()

        args, unknown = parser.parse_known_args()

        # Update config values with parsed arguments
        self.K = args.K
        self.KS = args.KS
        self.dirichlet_alpha = args.dirichlet_alpha
        self.emission_func = args.emission_func
        self.model = args.model
        self.sampler = args.sampler
