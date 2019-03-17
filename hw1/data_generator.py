import numpy as np
import pickle
import json
import os
import tensorflow as tf


class DataGenerator:
	def __init__(self, config):
		self.config = config
		# load data here
		self.load_data()

	def load_data(self):
		with open(self.config.data, 'rb') as f:
			d = pickle.load(f)
		features = d["observations"]
		labels = d["actions"]

		# save length of data
		self.length = len(labels)
		self.num_labels = labels.shape[-1]
		self.num_features = features.shape[-1]
		# save configs
		data_configs = {"length": self.length, "num_labels": self.num_labels, "num_features": self.num_features}
		print(data_configs)
		with open(os.path.join(self.config.summary_dir, "data_configs.json"), 'w') as outfile:
			json.dump(data_configs, outfile)

		labels = labels.reshape(self.length, self.num_labels)

		# create tensorflow dataset
		self.dataset = tf.data.Dataset.from_tensor_slices((features, labels))
		self.dataset = self.dataset.shuffle(buffer_size=10000)
		self.dataset = self.dataset.batch(self.config.batch_size)
		self.dataset = self.dataset.repeat(self.config.num_epochs)
		# create iterator
		self.iterator = self.dataset.make_initializable_iterator()
		self.x, self.y = self.iterator.get_next()
