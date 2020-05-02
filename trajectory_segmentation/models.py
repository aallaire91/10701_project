from sklearn import mixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import scale
from scipy.stats import anderson

import pyhsmm
from pyhsmm.util.text import progprint_xrange
import numpy as np
from matplotlib import pyplot as plt
from pyhsmm.basic.distributions import NegativeBinomialIntegerR2Duration
import autoregressive.models as m
import autoregressive.distributions as di
import copy




class HiddenSemiMarkovModel:

    def __init__(self):
        self.transitions = []
        self.model = []
        # internal variables not for outside reference
        self._demonstrations = []
        self._demonstration_sizes = []


    def add_demo(self, demonstration):
        demo_size = demonstration.shape

        self._demonstration_sizes.append(demo_size)
        self._demonstrations.append(demonstration)


    def fit(self):

        Nmax = 30

        # and some hyperparameters
        obs_dim = self._demonstrations[0].shape[1]

        obs_hypparams = {'mu_0': np.zeros(obs_dim),
                         'sigma_0': np.eye(obs_dim),
                         'kappa_0': .1,
                         'nu_0': obs_dim + 2}
        dur_hypparams = {'alpha_0': 30,
                         'beta_0': 2}

        obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
        dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

        posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
            alpha_a_0=1., alpha_b_0=.1,
            gamma_a_0=1., gamma_b_0=.1,
            init_state_concentration=10,
            obs_distns=obs_distns,
            dur_distns=dur_distns)

        for d in self._demonstrations:
            posteriormodel.add_data(d,trunc=400)  # duration truncation speeds things up when it's possible

        # for idx in progprint_xrange(300):
        #     posteriormodel.resample_model()

        # fig = plt.figure()
        models = []
        for idx in progprint_xrange(500):
            posteriormodel.resample_model()
            # if (idx + 1) % 50 == 0:
            #     plt.clf()
            #     posteriormodel.plot()
            #     plt.gcf().suptitle('HDP-HSMM sampled after %d iterations' % idx)
            #     plt.savefig('hsmm_iter/iter_%.3d.png' % idx)


        new_segments = []
        for i in range(0, len(self._demonstrations)):
            new_segments.append(find_transitions(posteriormodel.states_list[i].stateseq))

        self.transitions = new_segments
        self.model = posteriormodel

        return self.transitions



"""
Uses an off the shelf AutoregressiveMarkovModel
"""


class AutoregressiveMarkovModel:
    def __init__(self, lag=4, alpha=1, gamma=4, nu=2, init_state_concentration=1):
        self.transitions = []
        self.model = []
        self.lag = lag
        self.alpha = alpha
        self.nu = nu
        self.gamma = gamma
        # self.cap=cap
        self.init_state_concentration = init_state_concentration
        # internal variables not for outside reference
        self._demonstrations = []
        self._demonstration_sizes = []


    def add_demo(self, demonstration):
        demo_size = demonstration.shape

        self._demonstration_sizes.append(demo_size)
        self._demonstrations.append(demonstration)

    """
    Essentially taken from Matt Johnson's demo
    """

    def fit(self):


        p = self._demonstration_sizes[0][1]

        Nmax = 15#self._demonstration_sizes[0][0]//60
        self.nu =p+2

        affine = True
        nlags = self.lag
        obs_distns = [di.AutoRegression(nu_0=self.nu, S_0=np.eye(p), M_0=np.zeros((p, p + affine)),
            K_0=np.eye( p + affine), affine=affine) for state in range(int(Nmax))]
        dur_hypparams = {'alpha_0': 1,
                         'beta_0': 1}
        dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

        model = m.ARWeakLimitHDPHSMM(
            alpha_a_0=1., alpha_b_0=1,
            gamma_a_0=0.1, gamma_b_0=1,
            init_state_concentration=100,
            obs_distns=obs_distns,
            dur_distns=dur_distns)


        for d in self._demonstrations:
            model.add_data(d)

        # model.resample_model()

        for itr in progprint_xrange(400):
            model.resample_model()

        new_segments = []
        for i in range(0, len(self._demonstrations)):
            new_segments.append(find_transitions(model.states_list[i].stateseq))

        self.transitions = new_segments
        self.model = model
        return self.transitions



class GMeans:
    """strictness = how strict should the anderson-darling test for normality be
            0: not at all strict
            4: very strict
    """
    def __init__(self, min_obs = 30,max_depth=30, random_state=None, strictness=4):
        self.max_depth = max_depth
        self.random_state = random_state
        self.min_obs = min_obs
        if strictness not in range(5):
            raise ValueError("strictness parameter must be integer from 0 to 4")
        self.strictness = strictness
        self.stopping_criteria = []
        self.data = np.array([])
        self.data_index = np.array([])
        self.labels = np.array([])
        self.n_clusters = 0
        self.transitions = []
        self.model = []

        # internal variables not for outside reference
        self._demonstrations = []
        self._demonstration_sizes = []


    def _gaussianCheck(self, vector):
        """
        check whether a given input vector follows a gaussian distribution
        H0: vector is distributed gaussian
        H1: vector is not distributed gaussian
        """
        output = anderson(vector)

        if output[0] <= output[1][self.strictness]:
            return True
        else:
            return False

    def _recursiveClustering(self, data, depth, index):
        """
        recursively run kmeans with k=2 on your data until a max_depth is reached or we have
            gaussian clusters
        """
        depth += 1
        if depth == self.max_depth:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('max_depth')
            return

        km = MiniBatchKMeans(n_clusters=2, random_state=self.random_state)
        km.fit(data)

        centers = km.cluster_centers_
        v = centers[0] - centers[1]
        x_prime = scale(data.dot(v) / (v.dot(v)))
        gaussian = self._gaussianCheck(x_prime)

        # print gaussian
        if gaussian:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('gaussian')
            return

        labels = set(km.labels_)
        for k in labels:
            current_data = data[km.labels_ == k]

            if current_data.shape[0] <= self.min_obs:
                self.data_index[index[:, 0]] = index
                self.stopping_criteria.append('min_obs')
                return

            current_index = index[km.labels_ == k]
            current_index[:, 1] = np.random.randint(0, 100000)
            self._recursiveClustering(data=current_data, depth=depth, index=current_index)

    def add_demo(self, demonstration):
        demo_size = demonstration.shape

        self._demonstration_sizes.append(demo_size)
        self._demonstrations.append(demonstration)

    # set_trace()
    def fit(self):
        """
        fit the recursive clustering model to the data
        """
        for i in range(0,len(self._demonstrations)):

            if self.data.size == 0:
                self.data = self._demonstrations[i]
            else:
                self.data = np.vstack((self.data, self._demonstrations[i]))

        data_index = np.array([(i, False) for i in range(self.data.shape[0])])
        self.data_index = data_index

        self._recursiveClustering(data=self.data, depth=0, index=data_index)

        self.labels = self.data_index[:, 1]
        self.n_clusters = np.unique(self.labels).size
        return self.n_clusters


class TimeVaryingGaussianMixtureModel:

    def __init__(self, max_clusters=50, k=30, k_func=None,dp=False):
        self.max_clusters = max_clusters
        self.k_func = k_func
        self.k_fixed = k
        self.dp = dp

        self.data = np.array([])
        self.transitions = []
        self.model = []

        # internal variables not for outside reference
        self._demonstrations = []
        self._demonstration_sizes = []

    def add_demo(self, demonstration):
        demo_size = demonstration.shape

        self._demonstration_sizes.append(demo_size)
        self._demonstrations.append(demonstration)

    # this fits using the BIC, unless hard param is specified
    def fit(self):
        for i in range(0, len(self._demonstrations)):
            if self.data.size == 0:
                self.data = self._demonstrations[i]
            else:
                self.data = np.vstack((self.data, self._demonstrations[i]))

        self.labels = []
        self.model = []
        g = []
        if self.k_func is not None:
            k = self.k_func(self.data,self.max_clusters)
        else:
            k = self.k_fixed

        if self.dp:
            g = mixture.BayesianGaussianMixture(n_components=self.max_clusters, weight_concentration_prior_type="dirichlet_process",covariance_type='diag', max_iter=10000,weight_concentration_prior=0.001,tol=1e-7)
        else:

            g = mixture.GaussianMixture(n_components=k,covariance_type='diag', max_iter=10000,tol =5e-5)

        g.fit(self.data)
        self.model = g
        pred_labels = g.predict(self.data)
        new_segments = []
        start_i = 0
        for i in range(0, len(self._demonstrations)):
            end_i = start_i + int(self._demonstration_sizes[i][0])
            new_segments.append(find_transitions(pred_labels[start_i:end_i]))
            start_i = end_i
        self.transitions = new_segments
        return self.transitions


def k_bic(data,max_clusters):
    gmm_list = np.empty((0,2))
    for k in range(1, max_clusters):
        g = mixture.GaussianMixture(n_components=k)
        g.fit(data)
        gmm_list = np.vstack((gmm_list,(g.bic(data), g)))  # lower bic better

    best_k = np.argmin(gmm_list[:,0])
    return int(best_k+1)


def k_gmeans(data,max_segments):
    gmeans = GMeans()
    gmeans.add_demo(data)
    gmeans.fit()
    return int(gmeans.n_clusters)


# this finds the segment end points
def find_transitions(pred_labels):
    transitions = []
    seg_labels = []
    prev = -1
    for i, cur in enumerate(pred_labels):
        if (prev != cur) & (i != (len(pred_labels) - 1)):
            transitions.append(i)
            seg_labels.append(pred_labels[i])

        prev = cur

    transitions.append(pred_labels.size - 1)
    return [np.array(transitions), np.array(seg_labels)]
