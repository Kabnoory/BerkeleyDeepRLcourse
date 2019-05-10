import tensorflow as tf
from tqdm import tqdm
import numpy as np

class Trainer:
	def __init__(self, sess, model, config, logger=None):
		self.model = model
		self.config = config
		self.logger = logger
		self.sess = sess
		self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(self.init)
		self.num_epochs = 0

	def fit(self, data):
		# prepare training-data pipeline
		self.num_iter_per_epoch = (data.length // self.config.num_epochs) // self.config.batch_size
		self.handle = self.sess.run(data.iterator.string_handle())
		self.num_epochs += self.config.num_epochs
		self.sess.run(data.iterator.initializer)
		# train
		for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.num_epochs + 1, 1):
			self.train_epoch()
			self.sess.run(self.model.increment_cur_epoch_tensor)
		self.model.save(self.sess)

	def predict(self, observation):
		return self.sess.run(self.model.output, feed_dict={self.model.x: observation})

	def train_epoch(self):
		loop = tqdm(range(self.num_iter_per_epoch))
		losses = []
		for _ in loop:
			loss = self.train_step()
			losses.append(loss)
		loss = np.mean(losses)
		print("Epoch loss: ", loss)

		cur_it = self.model.global_step_tensor.eval(self.sess)
		summaries_dict = {
			'loss': loss
		}
		self.logger.summarize(cur_it, summaries_dict=summaries_dict)

	def train_step(self): 
		_, loss = self.sess.run([self.model.train_step, self.model.mse], 
								feed_dict={self.model.handle: self.handle})
		return loss
