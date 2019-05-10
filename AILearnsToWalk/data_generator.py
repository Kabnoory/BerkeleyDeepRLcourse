import numpy as np
import pickle
import json
import os
import tensorflow as tf


class DataGenerator:
	def __init__(self, config, raw=None):
		# load data here
		self.config = config
		if raw is not None:
			self.load_raw_data(raw)
		else:
			self.load_data_from_config()

	def load_data_from_config(self):
		with open(self.config.data, 'rb') as f:
			data = pickle.load(f)
		self.load_raw_data(data)

	def load_raw_data(self, data):
		features = data["observations"]
		labels = data["actions"]
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
		print("Creating tensorized dataset, might take a few moments...")
		self.dataset = tf.data.Dataset.from_tensor_slices((features, labels))
		self.dataset = self.dataset.shuffle(buffer_size=1000)
		self.dataset = self.dataset.batch(self.config.batch_size)
		self.dataset = self.dataset.repeat(self.config.num_epochs)
		# create iterator
		self.iterator = self.dataset.make_initializable_iterator()
