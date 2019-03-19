import tensorflow as tf


class Model:
	def __init__(self, config, num_labels=None, output_types=None, output_shapes=None, is_training=True):
		self.config = config
		self.is_training = is_training
		if is_training:
			if (not output_types) or (not output_shapes) or (not num_labels):
				print("ERROR: Missing required model training parameters")
				sys.exit(1)

			# handle constructions
			self.handle = tf.placeholder(tf.string, shape=[])
			self.iterator = tf.data.Iterator.from_string_handle(
				self.handle, output_types, output_shapes)
			self.x, self.y = self.iterator.get_next()
			self.num_labels = num_labels
		else:
			self.x = tf.placeholder(tf.float64, shape=[None, self.config.num_features])
			self.y = tf.placeholder(tf.float64, shape=[None, self.config.num_labels])
			self.num_labels = self.config.num_labels
		
		# init the global step
		self.init_global_step()
		# init the epoch counter
		self.init_cur_epoch()

		self.build_model()
		self.init_saver()


	def build_model(self):
		# network architectures
		d1 = tf.layers.dense(self.x, 128, activation=tf.nn.relu, name="dense1")
		d1 = tf.layers.dropout(d1, rate=0.2, training=self.is_training)
		d2 = tf.layers.dense(d1, 128, activation=tf.nn.relu, name="dense2")
		d2 = tf.layers.dropout(d2, rate=0.2, training=self.is_training)
		self.output = tf.layers.dense(d2, self.num_labels, name="dense3")

		with tf.name_scope("loss"):
			self.mse = tf.losses.mean_squared_error(labels=self.y, predictions=self.output)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.mse,
																						 global_step=self.global_step_tensor)


	# save function that saves the checkpoint in the path defined in the config file
	def save(self, sess):
		print("Saving model...")
		self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
		print("Model saved")

	# load latest checkpoint from the experiment path defined in the config file
	def load(self, sess):
		latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
		if latest_checkpoint:
			print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
			self.saver.restore(sess, latest_checkpoint)
			print("Model loaded")

	# just initialize a tensorflow variable to use it as epoch counter
	def init_cur_epoch(self):
		with tf.variable_scope('cur_epoch'):
			self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
			self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

	# just initialize a tensorflow variable to use it as global step counter
	def init_global_step(self):
		# DON'T forget to add the global step tensor to the tensorflow trainer
		with tf.variable_scope('global_step'):
			self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

	# initialize saver
	def init_saver(self):
		self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)