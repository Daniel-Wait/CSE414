import numpy as np
from gstuff import KMeans as km
import math
import matplotlib.pyplot as plt

class GMM:

	def __init__(self, k, tolerance): # this tolerance is for the log-likelihood (not the k means)

		# checks the validity of the k variable
		if not isinstance(k, int):
			raise TypeError("k must be an integer.")
		if (k < 2):
			raise ValueError("k must be larger than or equal to 2.")

		self.__k = k
		self.__tolerance = float(tolerance)
		self.__log_likelihood = -1e5  # this just initializes the log_likelihood



	def fit(self, X): # This methods takes row data

		# convert the data to column form
		X = X.T

		# initialise the variables
		self.__data = X
		self.__d = X.shape[0]
		self.__N = X.shape[1]

		# initialise with the KMean algorithm
		self.__get_k_means_parameters()

		while True:
			self.__calculate_responsibilities() # The E-STEP
			self.__calculate_parameters()       # The M-STEP

			#print(self.__log_likelihood)  #-- use this to see the likelihood converging

			self.__update_log_likelihood()

			if (self.__log_likelihood_converged()):
				break # breaks the while loop once the log-likelihood has converged


	def __get_k_means_parameters(self):

		# this makes use of my KMeans code
		clf = km.KMeans(self.__k, 1e-5)
		clf.cluster(self.__data.T)  # takes data in row form

		# extract the parameters from the KMeans object
		self.__means = clf.get_means()
		self.__sigmas = clf.get_covs()
		self.__priors = clf.get_priors()



	def __calculate_responsibilities(self):

		# initialise the responsibilities
		self.__responsibilities = np.zeros([self.__k, self.__N])

		# 
		for n in range(0, self.__N):

			# initialise the denominator
			denominator = 0

			for m in range(0, self.__k):

				denominator += self.__joint_mixture(m, n)

				# this is just the numerators, so far
				self.__responsibilities[m, n] = self.__joint_mixture(m, n)

			self.__responsibilities[:, n] /= denominator



	def __joint_mixture(self, m, n):

		# returns the product of the prior and the N(x|u, sigma), which is a type of likelihood
		return self.__priors[m] * self.__sub_likelihood(m, n)



	def __sub_likelihood(self, m, n):

		# get the factor in front of the gaussian pdf
		factor = float(1) / np.sqrt(np.linalg.det(2 * np.pi * self.__sigmas[m, :, :]))

		cx = self.__data[:, n].astype(np.float32)[:, np.newaxis]
		cx -= self.__means[:, m][:, np.newaxis]

		sigma_inverse = np.linalg.inv(self.__sigmas[m, :, :])

		# get the exponent part of the pdf
		exponent = math.exp(-0.5 * cx.T.dot(sigma_inverse.dot(cx)))

		return factor * exponent



	def __calculate_parameters(self):
		self.__calculate_cluster_counts()
		self.__calculate_means()
		self.__calculate_sigmas()
		self.__calculate_priors()



	def __calculate_cluster_counts(self):

		# initialise the cluster counts (these are the total elements in a given cluster)
		self.__cluster_counts = np.zeros(self.__k)

		for m in range(0, self.__k):
			for n in range(0, self.__N):
				self.__cluster_counts[m] += self.__responsibilities[m, n]



	def __calculate_means(self):

		# initialises the means
		self.__means = np.zeros([self.__d, self.__k])

		for m in range(0, self.__k):
			for n in range(0, self.__N):
				x = self.__data[:, n].astype(np.float32)
				mean_layer = self.__responsibilities[m, n] * x

				for l in range(0, self.__d):
					self.__means[l, m] += mean_layer[l]

			# normalises the means
			self.__means[:, m] /= self.__cluster_counts[m]



	def __calculate_sigmas(self):

		# initialise the sigmas/ covariances
		self.__sigmas = np.zeros([self.__k, self.__d, self.__d])

		for m in range(0, self.__k):
			for n in range(0, self.__N):
				cx = self.__data[:, n].astype(np.float32)[:, np.newaxis]
				cx -= self.__means[:, m][:, np.newaxis]

				sigma_layer = self.__responsibilities[m, n] * cx.dot(cx.T)

				for l in range(0, self.__d):
					for q in range(0, self.__d):
						self.__sigmas[m, l, q] += sigma_layer[l, q]

			# normalises the sigmas
			self.__sigmas[m, :, :] /= self.__cluster_counts[m]



	def __calculate_priors(self):

		# initialise the priors
		self.__priors = np.zeros(self.__k)

		for m in range(0, self.__k):
			self.__priors[m] = self.__cluster_counts[m] / float(self.__N)



	def __update_log_likelihood(self):
		self.__old_log_likelihood = self.__log_likelihood
		self.__calculate_log_likelihood()



	def __calculate_log_likelihood(self):
		self.__log_likelihood = 0

		for n in range(0, self.__N):
			likelihood_layer = 0
			for m in range(0, self.__k):
				likelihood_layer += self.__joint_mixture(m, n)
				self.__log_likelihood += math.log(likelihood_layer)



	def __log_likelihood_converged(self):

		# calculate the log likelihood step/error
		log_likelihood_error = math.fabs(self.__log_likelihood - self.__old_log_likelihood)

		# check if the likelihood step is smaller than the specified tolerance
		return (log_likelihood_error < self.__tolerance)


	# Thanks to Alexamder Van Zyl, for showing me how to do this:
	def draw(self):
		colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'lime', 'goldenrod']
		plt.figure()

		for n in range(0, self.__N):
			for m in range(0, self.__k):
				plt.scatter(self.__data[0, n], self.__data[1, n], marker='o', 
					color=colors[m], alpha=self.__responsibilities[m, n]/4)




	# Accessor Methods
	def get_cluster_counts(self):
		return self.__cluster_counts

	def get_means(self):
		return self.__means

	def get_covs(self):
		return self.__sigmas

	def get_responsibilities(self):
		return self.__responsibilities