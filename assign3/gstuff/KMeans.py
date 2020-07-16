import numpy as np

class KMeans:

	#constructor with implicit attributes
	def __init__(self, k, tolerance):

		# checks if k is an integer that is larger than 1
		if not isinstance(k, int):
			raise TypeError("k must be an integer!")
		if (k < 2):
			raise ValueError("k must be larger than or equal to 2!")

		# initialises the values
		self.__k_total = k
		self.__tolerance = float(tolerance)



	def cluster(self, X): # This function takes the X data in row form
		
		X = X.T # converts the data to column form

		# save the data and dimensions
		self.__data = X
		self.__d = X.shape[0]
		self.__N = X.shape[1]
		self.__labels = np.ones([1, self.__N])
		self.__means = np.zeros([self.__d, 1])

		# add one cluster at a time
		for m in range(1, self.__k_total):

			# set the current number of clusters
			self.__k = m
			self.__calculate_sigmas()

			# find the index of the cluster with the maximum spread
			q = self.__get_max_sigma_index()

			# split the cluster with the maximum spread
			self.__split_cluster(q)

			# flip between classification and finding the means until it converges
			while True:
				self.__classify()
				self.__update_means()

				if (self.__means_converged()):
					break

		self.__calculate_all_parameters()

			

	def __calculate_sigmas(self):

		# initialise the sigmas (covariances)
		self.__sigmas = np.zeros([self.__k, self.__d, self.__d])

		for m in range(0, self.__k):
			
			count = 0

			for n in range(0, self.__N):

				# check if the current value is part of the current class
				if (self.__labels[m, n] == 1):

					# find the centered value
					cx = self.__data[:, n].astype(np.float32)[:, np.newaxis]
					cx -= self.__means[:, m][:, np.newaxis]

					# get one layer of the sigma matrix
					sigma_layer = cx.dot(cx.T)

					# add the layer to the sigma matrix
					for l in range(0, self.__d):
						for q in range(0, self.__d):
							self.__sigmas[m, l, q] += sigma_layer[l, q]

					count += 1

			# normalise the sigma matrix
			for l in range(0, self.__d):
				for q in range(0, self.__d):
					self.__sigmas[m, l, q] /= count



	def __get_max_sigma_index(self):

		# initialise the mutliplicative traces (prod)
		determinants = np.ones([1, self.__k]) # notice that is initialised with ones, not zeros

		for m in range(0, self.__k):

			# find the eigenvalues
			(eig_vals, eig_vecs) = np.linalg.eigh(self.__sigmas[m, :, :])
			eigs = eig_vals

			# find the determinants by multiplying the eigenvalues together
			for l in range(0, self.__d):
				determinants[0, m] *= eigs[l]

		# the determinant of the diagonal matrix gives a sense of its spread
		# return the maximum spread's argument
		return np.argmax(determinants, axis=1)



	def __split_cluster(self, q):

		# find the maximum and minimum positions within a given cluster
		x_min = self.__boundary_min(q)
		x_max = self.__boundary_max(q)

		# initialise the two new means of the cluster to be split
		mean_1 = np.zeros([self.__d, 1])
		mean_2 = np.zeros([self.__d, 1])

		# randomize
		np.random.seed(self.__k + 1)

		# find random positions for the two means within the cluster to be split
		for l in range(0, self.__d):
			mean_1[l, 0] = np.random.uniform(x_min[l, 0], x_max[l, 0])
			mean_2[l, 0] = np.random.uniform(x_min[l, 0], x_max[l, 0])

		# replace the mean of the cluster to be split with mean_1
		for l in range(0, self.__d):
			self.__means[l, q] = mean_1[l, 0]

		# append mean_2 to the matrix of means
		self.__means = np.hstack((self.__means, mean_2))

		# increment the number of clusters
		self.__k += 1



	def __boundary_min(self, q):

		# initialise the minimum position
		min = None

		# set the minimum to the first value in the q'th cluster
		for n in range(0, self.__N):

			if (self.__labels[q, n] == 1):
				min = self.__data[:, n].astype(np.float32)[:, np.newaxis]
				break

		# change the dimensions of the minimum position, if any smaller dimension is found
		# from data within the same cluster
		for n in range(0, self.__N):

			if (self.__labels[q, n] == 1):
				x = self.__data[:, n].astype(np.float32)

				for l in range(0, self.__d):
					if (x[l] < min[l]):

						min[l] = x[l]

		# return the minimum position
		return min



	def __boundary_max(self, q):

		# initialise the maximum position
		max = None

		# set the maximum to the first value in the q'th cluster
		for n in range(0, self.__N):
			if (self.__labels[q, n] == 1):
				max = self.__data[:, n].astype(np.float32)[:, np.newaxis]
				break

		# change the dimensions of the maximum position, if any larger dimension is found
		# from data within the same cluster
		for n in range(0, self.__N):

			if (self.__labels[q, n] == 1):
				x = self.__data[:, n].astype(np.float32)

				for l in range(0, self.__d):
					if (x[l] > max[l]):

						max[l] = x[l]

		# return the maximum position
		return max



	def __classify(self):

		# find the distances between all the data and the cluster means
		dist = self.__data.T[:, :, np.newaxis] - self.__means[np.newaxis, :, :]

		# find the errors (norms of distances) of the data relative to the means
		errors = np.linalg.norm(dist, axis=1)

		# find the label indices from the errors (the minimum error = closests mean)
		label_indices = np.argmin(errors, axis=1)

		# initialise the labels
		self.__labels = np.zeros([self.__k, self.__N])

		# find the 1-of-k coding labels from the label_indices
		for n in range(0, self.__N):
			self.__labels[label_indices[n], n] = 1



	def __update_means(self):

		# save the means as old_means
		self.__old_means = self.__means

		self.__calculate_means()



	def __calculate_means(self):

		# initialise the new means
		self.__means = np.zeros([self.__d, self.__k])

		for m in range(0, self.__k):
			count = 0

			for n in range(0, self.__N):

				# add all the values in the current cluster to the mean
				if (self.__labels[m, n] == 1):
					x = self.__data[:, n].astype(np.float32)[:, np.newaxis]

					for l in range(0, self.__d):
						self.__means[l, m] += x[l]

					count += 1

			# normalise the mean
			self.__means[:, m] /= count



	def __means_converged(self):

		# find the step vectors of the means
		mean_steps = self.__means - self.__old_means

		# see how far they stepped
		mean_errors = np.linalg.norm(mean_steps, axis=0)

		converged = True
		for m in range(0, self.__k):

			# check if the steps are smaller than the tollerance
			if (mean_errors[m] > self.__tolerance):
				converged = False

		return converged



	def __calculate_all_parameters(self):
		self.__calculate_means()
		self.__calculate_sigmas()

		self.__calculate_class_labels()
		self.__calculate_class_counts()
		self.__calculate_priors()



	def __calculate_class_labels(self):

		self.__class_labels = [0 for n in range(0, self.__N)]

		# converts from 1-of-k coding labels to regular class labels
		for n in range(0, self.__N):
			for m in range(0, self.__k):
				if (self.__labels[m, n] == 1):
					self.__class_labels[n] = m



	def __calculate_class_counts(self):
		self.__class_counts = np.zeros(self.__k).astype(np.int32)

		for m in range(0, self.__k):
			for n in range(0, self.__N):
				if (self.__labels[m, n] == 1):
					self.__class_counts[m] += 1



	def __calculate_priors(self):
		self.__priors = np.zeros(self.__k).astype(np.float32)

		for m in range(0, self.__k):
			self.__priors[m] = self.__class_counts[m] / self.__N


	# Accessor methods
	def get_labels(self):
		return self.__labels

	def get_class_labels(self):
		return self.__class_labels

	def get_means(self):
		return self.__means

	def get_covs(self):
		return self.__sigmas

	def get_class_counts(self):
		return self.__class_counts

	def get_priors(self):
		return self.__priors
