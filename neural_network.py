import numpy as np, pickle

class NN:
	def __init__(self, input_len, hidden_layers, output_len): # I think I still need to add biases
		# print(hidden_layers)
		self.weights = []
		self.biases = []
		if hidden_layers:
			input_size = (input_len, hidden_layers[0])
			output_size = (hidden_layers[-1], output_len)

			# self.biases.append(np.random.randn(hidden_layers[0]))

			# self.weights.append(np.random.rand(*input_size))
			self.weights.append(np.random.randn(*input_size))

			for i, hidden_layer in enumerate(hidden_layers[:-1]):
				next_size = (hidden_layer, hidden_layers[i+1])
				# self.weights.append(np.random.rand(*next_size))
				self.weights.append(np.random.randn(*next_size))
				self.biases.append(np.random.randn(hidden_layer))
				# print(hidden_layer)
				# print(len(self.biases[i]))

			# self.weights.append(np.random.rand(*output_size))
			self.weights.append(np.random.randn(*output_size))
			self.biases.append(np.random.randn(hidden_layers[-1]))
			self.biases.append(np.random.randn(output_len))
		else:
			input_size = (input_len, output_len)
			# self.weights.append(np.random.rand(*input_size))
			self.weights.append(np.random.randn(*input_size))

		# print(self.weights)

	def forward(self, x): # I think I still need to add biases
		z = np.array(x)

		# for i, weight in enumerate(self.weights):
		for i, weight in enumerate(self.weights):
			# print("layer {}".format(i))
			# z = self.activation(weight.T.dot(z))
			# print(i, weight.shape, len(self.biases[i]))
			z = self.activation(z.dot(weight) + self.biases[i])
			# print(z)
			# print(z.shape)
			# z = weight.T.dot(z)

		return z
		# return self.output_layer(z)

	def activation(self, z):
		# # linear
		# return z

		# # relu
		# return np.maximum(np.zeros(z.shape), z)

		# sigmoid
		return (1 + np.exp(-z))**(-1)

	def output_layer(self, z):
		# soft max
		# print(z)
		# max_z = np.amin(z) # this prevents overflow!
		max_z = np.amax(z, axis=1)[:, np.newaxis] # this prevents overflow!
		# print(z - max_z)
		scores = np.exp(z - max_z) # this prevents overflow!
		# scores = np.array(z)
		# print(scores)
		return scores / np.sum(scores)
		# wait! this causes a problem! This means that one neuron will always output 0!!!
		# thus, if you only have 2 outputs, one will always be 0 and another will always be 1!!!
		# WTF?!

	# def loss(self, X, y):
	# 	losses = []
	# 	for i, x in enumerate(X):
	# 		losses.append(L_i(self.forward(x), y[i])) # if not one-hot
	# 		# losses.append(L_i(self.forward(x), [y[i] == 1])) # you can do with dot prod instead of y == 1 (when you vectorize)

	# 	return sum(losses)

	# def L_i(self, scores, y_i): # loss for observation i?
	# 	scores_ = scores - np.amax(scores)
	# 	scores_ = np.exp(scores_)
	# 	return -np.log(scores_[y_i]/np.sum(scores_))

	def predict(self, x):
		scores = self.forward(x)
		probs = self.output_layer(scores)
		label = np.argmax(probs, axis=1)
		return label

	def save(self, file_name):
		pickle.dump([self.weights, self.biases], open(file_name, "wb"))

	def load(self, file_name):
		self.weights, self.biases = pickle.load(open(file_name, "rb"))

if __name__ == "__main__":
	x_n = 10
	n = NN(x_n, [2**x for x in np.arange(1, 15)[::-3]], 10)
	# print(n.output_layer(n.forward(np.random.randn(x_n))))
	print(n.predict(np.random.randn(2, x_n)))
	# print(n.predict(np.random.randn(1000, x_n)))
