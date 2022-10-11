import os
import numpy as np
import pickle as pkl

from sklearn.cluster import KMeans
from scipy.stats import norm
from matplotlib import pyplot as plt

class NDB:
    def __init__(self, training_data=None, number_of_bins=100, significance_level=0.05, z_threshold=None,
                 whitening=False, max_dims=None, cache_folder=None):
        """Number of statistically-different bins 

        Args:
            training_data (numpy.array, optional): images to train the K-means partition. Defaults to None.
            number_of_bins (int, optional): number of bins of the K-means model. Defaults to 100.
            significance_level (float, optional): threshold indicating statistically different bins. Defaults to 0.05.
            z_threshold (float, optional):  test statistic threshold statistically different bins. Defaults to None.
            whitening (bool, optional): if True, input is whitened. Defaults to False.
            max_dims (int, optional): Maximum number of pixels used to fit K-means. Defaults to None.
            cache_folder (str, optional): folder to save pre-trained K-means results. Defaults to None.
        """
        self.number_of_bins = number_of_bins  
        self.significance_level = significance_level
        self.z_threshold = z_threshold  
        self.whitening = whitening
        self.max_dims = max_dims
        self.cache_folder = cache_folder
        
        self.ndb_eps = 1e-6  # 
        self.training_mean = 0.0
        self.training_std = 1.0
        
        self.bin_centers = None
        self.bin_proportions = None
        self.ref_sample_size = None
        self.used_d_indices = None
        self.results_file = None
        self.test_name = 'ndb_{}_bins_{}'.format(self.number_of_bins, 'whiten' if self.whitening else 'orig')
        self.cached_results = {}
        
        # Load from a cache folder where we stored the bin calculation
        if self.cache_folder:
            self.results_file = os.path.join(cache_folder, self.test_name+'_ndb_results.pkl')
            # If results exist, you open the existing files
            if os.path.isfile(self.results_file):
                print('Loading previous results from', self.results_file, ':')
                self.cached_results = pkl.load(open(self.results_file, 'rb'))
        
        # Create a folder where we save the results and create bins
        if training_data is not None or cache_folder is not None:
                bins_file = None
                if cache_folder:
                    os.makedirs(cache_folder, exist_ok=True)
                    bins_file = os.path.join(cache_folder, self.test_name+'.pkl')
                self.construct_bins(training_data, bins_file)

    def construct_bins(self, training_samples, bins_file):
        """Construct the bins via the K-means algorithm 

        Args:
            training_samples (numpy.array): images to train the K-means dataset 
            bins_file (str): the path to the pre-trained K-means output 
        """
        n, d = training_samples.shape
        # Bins define resolution 
        k = self.number_of_bins
        if self.whitening:
            self.training_mean = np.mean(training_samples, axis=0)
            self.training_std = np.std(training_samples, axis=0) + self.ndb_eps
        
        # To run faster, perform binning on sampled data dimension 
        if self.max_dims is None and d > 1000:
            self.max_dims = d//6

        whitened_samples = (training_samples-self.training_mean)/self.training_std
        d_used = d if self.max_dims is None else min(d, self.max_dims)
        # Randomly pick pixel dimensions 
        self.used_d_indices = np.random.choice(d, d_used, replace=False)

        # Run the K-means algorithm on the training data using K-clusters 
        clusters = KMeans(n_clusters=k, max_iter=100).fit(whitened_samples[:, self.used_d_indices])
        
        # Compute centroid for each category
        bin_centers = np.zeros([k, d])
        for i in range(k):
            bin_centers[i, :] = np.mean(whitened_samples[clusters.labels_ == i, :], axis=0)

        # Organize bins by size
        label_vals, label_counts = np.unique(clusters.labels_, return_counts=True)
        bin_order = np.argsort(-label_counts)
        # Save centres and proportions as attributes
        self.bin_proportions = label_counts[bin_order] / np.sum(label_counts)
        self.bin_centers = bin_centers[bin_order, :]
        self.ref_sample_size = n

    def evaluate(self, query_samples, model_label=None):
        """Assign each sample to the nearest bin center (in terms of L2) and calculate the NDB

        Args:
            query_samples (numpy.array): m samples of dimension d
            model_label (str, optional):  optional label string for the evaluated model. Defaults to None.

        Returns:
            dict: results dictionary containing NDB and JS scores
        """
    
        n = query_samples.shape[0]
        query_bin_proportions, query_bin_assignments = self.__calculate_bin_proportions(query_samples)

        # Compute the two-sample test 
        different_bins = NDB.two_proportions_z_test(self.bin_proportions, 
                                                    self.ref_sample_size, 
                                                    query_bin_proportions,
                                                    n, significance_level=self.significance_level,
                                                    z_threshold=self.z_threshold)
        
        # Count how many bins are significant 
        ndb = np.count_nonzero(different_bins)
        js = NDB.jensen_shannon_divergence(self.bin_proportions, query_bin_proportions)
        results = {'NDB': ndb,
                   'JS': js,
                   'Proportions': query_bin_proportions,
                   'N': n,
                   'Bin-Assignment': query_bin_assignments,
                   'Different-Bins': different_bins}

        if model_label:
            print('Results for {} samples from {}: '.format(n, model_label), end='')
            self.cached_results[model_label] = results
            if self.results_file:
                pkl.dump(self.cached_results, open(self.results_file, 'wb'))

        return results


    def __calculate_bin_proportions(self, samples):
        """Compute the bin proportions in the query samples 

        Args:
            samples (numpy.array): Query sample with m samples and d dimensions

        Returns:
            tuple: probabilities and cluster assignment for each query sample 
        """
        
        if self.bin_centers is None:
            print('First run construct_bins on samples from the reference training data')
        assert samples.shape[1] == self.bin_centers.shape[1]
        
        # Get sample and dimensions size 
        n, d = samples.shape
        k = self.bin_centers.shape[0]  # Number of centres 
        D = np.zeros([n, k], dtype=samples.dtype)

        # Whiten sample with statistics from the training set 
        whitened_samples = (samples-self.training_mean)/self.training_std
        for i in range(k):
            # For each sample you measure its distance from all the kmeans centres 
            D[:, i] = np.linalg.norm(whitened_samples[:, self.used_d_indices] - self.bin_centers[i, self.used_d_indices],
                                     ord=2, axis=1)
            
        # Assign each observation to a class 
        labels = np.argmin(D, axis=1)
        probs = np.zeros([k])
        label_vals, label_counts = np.unique(labels, return_counts=True)
        # Get the label probabilities in the test set 
        probs[label_vals] = label_counts / n
        return probs, labels

    def __read_from_bins_file(self, bins_file):
        """Read the results of K-means from saved file

        Args:
            bins_file (str): the name of the file to read 

        Returns:
            bool: True if the file was found and loaded, false otherwise
        """
        if bins_file and os.path.isfile(bins_file):
            print('Loading binning results from', bins_file)
            bins_data = pkl.load(open(bins_file,'rb'))
            self.bin_proportions = bins_data['proportions']
            self.bin_centers = bins_data['centers']
            self.ref_sample_size = bins_data['n']
            self.training_mean = bins_data['mean']
            self.training_std = bins_data['std']
            self.used_d_indices = bins_data['d_indices']
            return True
        return False

    def __write_to_bins_file(self, bins_file):
        """Save the results of K-means onto the disk 

        Args:
            bins_file (str): the name of the file to read 
        """
        if bins_file:
            print('Caching binning results to', bins_file)
            bins_data = {'proportions': self.bin_proportions,
                         'centers': self.bin_centers,
                         'n': self.ref_sample_size,
                         'mean': self.training_mean,
                         'std': self.training_std,
                         'd_indices': self.used_d_indices}
            pkl.dump(bins_data, open(bins_file, 'wb'))

    @staticmethod
    def two_proportions_z_test(p1, n1, p2, n2, significance_level, z_threshold=None):
        """Kolmogorov-Smirnov two-sample test 

        Args:
            p1 (numpy.array): distribution in the partition of dataset 1
            n1 (int): number of observations in dataset 1
            p2 (numpy.array): distribution in the partition of dataset 2
            n2 (int): number of observations in dataset 2
            significance_level (float): threshold indicating statistically different bins. 
            z_threshold (float, optional):  test statistic threshold statistically different bins. Defaults to None.

        Returns:
            numpy.array: array of booleans representing significance of differential filling for each bin 
        """
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
        z = (p1 - p2) / se
        # Either define differential occupation based on test statistic or p-value
        if z_threshold is not None:
            return abs(z) > z_threshold
        # Perform 2-tail test
        p_values = 2.0 * norm.cdf(-1.0 * np.abs(z))    
        return p_values < significance_level

    @staticmethod
    def jensen_shannon_divergence(p, q):
        """Calculates the symmetric Jensen–Shannon divergence between the two PDFs

        Args:
            p (numpy.array): distribution across partition of dataset 1
            q (numpy.array): distribution across partition of dataset 2

        Returns:
            float: Jensen-Shannon divergence
        """
        m = (p + q) * 0.5
        return 0.5 * (NDB.kl_divergence(p, m) + NDB.kl_divergence(q, m))

    @staticmethod
    def kl_divergence(p, q):
        """Kullback-Leibler divergence between two 1-D distributions 

        Args:
            p (numpy.array): distribution across partition of dataset 1
            q (numpy.array): distribution across partition of dataset 2

        Returns:
            float: Kullback-Leibler divergence between p and q
        """
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(q))
        assert not np.any(np.logical_and(p != 0, q == 0))

        p_pos = (p > 0)
        return np.sum(p[p_pos] * np.log(p[p_pos] / q[p_pos]))
