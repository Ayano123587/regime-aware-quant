"""Diagonal Gaussian HMM with log-domain Baum-Welch and causal filtering.

Training uses smoothing within training history only. Forecast-time probabilities
use the forward filter, never Viterbi or forward-backward on held-out history.
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


class GaussianHMM:
    def __init__(self, states=3, iterations=60, seed=42):
        if states < 2 or iterations < 1:
            raise ValueError("Require >=2 states and positive iterations")
        self.states, self.iterations, self.seed = states, iterations, seed

    def emission(self, x):
        delta = np.asarray(x)[:, None, :] - self.means[None, :, :]
        return -.5 * (np.log(2*np.pi*self.variances).sum(axis=1)[None, :]
                      + (delta**2 / self.variances[None, :, :]).sum(axis=2))

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or len(x) < self.states*5 or not np.isfinite(x).all():
            raise ValueError("Need finite 2D observations and >=5 rows per state")
        self.center = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-6)
        z = (x-self.center)/self.scale
        labels = KMeans(self.states, n_init=5, random_state=self.seed).fit_predict(z)
        self.means = np.array([z[labels == k].mean(axis=0) for k in range(self.states)])
        self.variances = np.array([np.maximum(z[labels == k].var(axis=0), .05) for k in range(self.states)])
        self.initial = np.full(self.states, 1/self.states)
        self.transition = np.full((self.states, self.states), .05/(self.states-1))
        np.fill_diagonal(self.transition, .95)
        self.log_likelihoods = []
        for _ in range(self.iterations):
            emission = self.emission(z)
            log_a = np.log(self.transition)
            alpha = np.empty_like(emission)
            alpha[0] = np.log(self.initial) + emission[0]
            for t in range(1, len(z)):
                alpha[t] = emission[t] + logsumexp(alpha[t-1][:, None]+log_a, axis=0)
            ll = logsumexp(alpha[-1])
            beta = np.zeros_like(emission)
            for t in range(len(z)-2, -1, -1):
                beta[t] = logsumexp(log_a + emission[t+1][None, :] + beta[t+1][None, :], axis=1)
            gamma = np.exp(alpha + beta - ll)
            gamma /= gamma.sum(axis=1, keepdims=True)
            counts = np.zeros_like(self.transition)
            for t in range(len(z)-1):
                counts += np.exp(alpha[t][:, None]+log_a+emission[t+1][None, :]+beta[t+1][None, :]-ll)
            self.transition = counts + 1e-4
            self.transition /= self.transition.sum(axis=1, keepdims=True)
            self.initial = np.maximum(gamma[0], 1e-8)
            self.initial /= self.initial.sum()
            weight = np.maximum(gamma.sum(axis=0), 1e-8)
            self.means = gamma.T @ z / weight[:, None]
            self.variances = np.maximum((gamma.T @ (z*z))/weight[:, None] - self.means**2, .01)
            self.log_likelihoods.append(float(ll))
            if len(self.log_likelihoods) > 1 and abs(ll-self.log_likelihoods[-2]) < 1e-4:
                break
        return self

    def filter(self, x, previous=None):
        z = (np.asarray(x, dtype=float)-self.center)/self.scale
        if z.ndim != 2 or not np.isfinite(z).all():
            raise ValueError("Need finite 2D observations")
        if not len(z):
            return np.empty((0, self.states))
        emission = self.emission(z)
        posterior = None if previous is None else np.asarray(previous, dtype=float)
        output = []
        for row in emission:
            prior = self.initial if posterior is None else posterior @ self.transition
            log_p = np.log(np.maximum(prior, 1e-300))+row
            posterior = np.exp(log_p-logsumexp(log_p))
            output.append(posterior.copy())
        return np.asarray(output)
