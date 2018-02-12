import tensorflow as tf
import os
import numpy as np
from PIL import Image, ImageDraw
import random
import math

flags = tf.app.flags
flags.DEFINE_boolean("train_mode", False, "train_mode")
flags.DEFINE_boolean("using_batch_norm", False, "using_batch_norm")
flags.DEFINE_float("prob_loss_weight", 1.0, "prob_loss_weight")
flags.DEFINE_float("coor_loss_weight", 1000.0, "coor_loss_weight")
flags.DEFINE_float("size_loss_weight", 1000.0, "size_loss_weight")
flags.DEFINE_float("valid_box_prob_weight", 100.0, "valid_box_prob_weight")
flags.DEFINE_integer("train_batch_size", 2, "train_batch_size")
flags.DEFINE_integer("anchor_box_amt", 5, "anchor_box_amt")
flags.DEFINE_float("iou_thresh", 0.7, "iou_thresh")
flags.DEFINE_float("valid_prob_thresh", 0.95, "valid_prob_thresh")
flags.DEFINE_float("learning_rate", 0.00002, "Learning rate of for d network [0.0001]")
flags.DEFINE_integer("max_learning_step", 10000, "max_learning_step")
flags.DEFINE_integer("input_img_width", 640, "input_img_width")
flags.DEFINE_integer("input_img_height", 384, "input_img_height")
flags.DEFINE_integer("session_save_step", 5, "session_save_step")
flags.DEFINE_integer("output_cell_shape_width", 10, "output_cell_shape_width")
flags.DEFINE_integer("output_cell_shape_height", 6, "output_cell_shape_height")
flags.DEFINE_string("check_point_load_file", "", "should be empty if using the last check point")
flags.DEFINE_string("check_point_dir", "ckpt", "check_point_dir")
flags.DEFINE_string("graph_dir", "graph", "graph_dir")
flags.DEFINE_string("train_img_dir", "data/train/image", "train_img_dir")
flags.DEFINE_string("train_ground_truth_dir", "data/train/ground_truth", "train_ground_truth_dir")
flags.DEFINE_string("test_img_dir", "data/test/image", "test_img_dir")
flags.DEFINE_string("test_ground_truth_dir", "data/test/ground_truth", "test_ground_truth_dir")
flags.DEFINE_string("image_file_extension", ".png", "image_file_extension")
flags.DEFINE_string("ground_truth_file_extension", ".txt", "ground_truth_file_extension")
FLAGS = flags.FLAGS

def weight_variable(shape):
	tmp = tf.truncated_normal(shape, stddev=0.06);
	return tf.Variable(tmp)

def bias_variable(shape):
	tmp = tf.constant(0.1, shape=shape)
	return tf.Variable(tmp)
	
def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')

def max_pool_2x2(x, stride):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,stride,stride,1], padding='SAME')
	
def max_pool_3x3(x, stride):
	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,stride,stride,1], padding='SAME')
	
def avg_pool(x, height, width):
	shape = tf.shape(x)
	return tf.nn.avg_pool(x, ksize=[1,height,width,1], strides=[1,1,1,1], padding='VALID')

def leaky(x, k1=0.1, k2=0.5):
	return tf.maximum(k1*x, k2*x)

def batch_norm(inputs, is_training, is_conv_out=True, decay = 0.999):
	scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

	if is_training:
		if is_conv_out:
			batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
		else:
			batch_mean, batch_var = tf.nn.moments(inputs,[0])   

		train_mean = tf.assign(pop_mean,
								pop_mean * decay + batch_mean * (1 - decay))
		train_var = tf.assign(pop_var,
								pop_var * decay + batch_var * (1 - decay))
		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs,
				batch_mean, batch_var, beta, scale, 0.001)
	else:
		return tf.nn.batch_normalization(inputs,
			pop_mean, pop_var, beta, scale, 0.001)

class PD_network:
	def __init__(self):
		self.conv1 = weight_variable([3, 3, 3, 64])
		self.bias1 = bias_variable([64])
		self.conv2 = weight_variable([3, 3, 64, 192])
		self.bias2 = bias_variable([192])
		self.subnet_vars1 = self._subnet_variables(192, 64, 96, 128, 16, 32, 32)
		self.subnet_vars2 = self._subnet_variables(256, 128, 128, 192, 32, 96, 64)
		self.subnet_vars3 = self._subnet_variables(480, 192, 96, 208, 16, 48, 64)
		self.subnet_vars4 = self._subnet_variables(512, 160, 112, 224, 24, 64, 64)
		self.subnet_vars5 = self._subnet_variables(512, 128, 128, 256, 24, 64, 64)
		self.subnet_vars6 = self._subnet_variables(512, 112, 144, 288, 32, 64, 64)
		self.subnet_vars7 = self._subnet_variables(528, 256, 160, 320, 32, 128, 128)
		self.subnet_vars8 = self._subnet_variables(832, 384, 192, 384, 48, 128, 128)
		self.conv_prob = weight_variable([1, 1, 1024, 1*FLAGS.anchor_box_amt])
		self.bias_prob = bias_variable([1*FLAGS.anchor_box_amt])
		self.conv_center = weight_variable([1, 1, 1024, 2*FLAGS.anchor_box_amt])
		self.bias_center = bias_variable([2*FLAGS.anchor_box_amt])
		self.conv_size = weight_variable([1, 1, 1024, 2*FLAGS.anchor_box_amt])
		self.bias_size = bias_variable([2*FLAGS.anchor_box_amt])
		self.saver = tf.train.Saver()
		self.anchor_box_size = get_anchor_box_size()
		self.cell_shape = (FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width)
		self.cell_size = get_cell_size()
		self.cell_left_top = get_output_cell_left_top()
				
	def _subnet_variables(self, depth_in, depth_1x1, depth_3x3_red, depth_3x3, depth_5x5_red, depth_5x5, depth_pool):
		conv_single = weight_variable([1, 1, depth_in, depth_1x1])
		bias_single = bias_variable([depth_1x1])
		
		conv_3x3_red = weight_variable([1, 1, depth_in, depth_3x3_red])
		bias_3x3_red = bias_variable([depth_3x3_red])
		
		conv_3x3 = weight_variable([3, 3, depth_3x3_red, depth_3x3])
		bias_3x3 = bias_variable([depth_3x3])
		
		conv_5x5_red = weight_variable([1, 1, depth_in, depth_5x5_red])
		bias_5x5_red = bias_variable([depth_5x5_red])
		
		conv_5x5 = weight_variable([5, 5, depth_5x5_red, depth_5x5])
		bias_5x5 = bias_variable([depth_5x5])
		
		conv_pool = weight_variable([1, 1, depth_in, depth_pool])
		bias_pool = bias_variable([depth_pool])
		
		return [conv_single, bias_single, conv_3x3_red, bias_3x3_red, conv_3x3, bias_3x3, conv_5x5_red, bias_5x5_red, conv_5x5, bias_5x5, conv_pool, bias_pool]

	def _subnetwork(self, x, subnet_vars):
		y1 = leaky(conv2d(x, subnet_vars[0], 1)+subnet_vars[1])
		y2 = leaky(conv2d(x, subnet_vars[2], 1)+subnet_vars[3])
		y2 = leaky(conv2d(y2, subnet_vars[4], 1)+subnet_vars[5])
		y3 = leaky(conv2d(x, subnet_vars[6], 1)+subnet_vars[7])
		y3 = leaky(conv2d(y3, subnet_vars[8], 1)+subnet_vars[9])
		y4 = max_pool_3x3(x, 1)
		y4 = leaky(conv2d(y4, subnet_vars[10], 1)+subnet_vars[11])
		y = tf.concat([y1, y2, y3, y4], 3)
		return y
	
	def _subnetwork_bn(self, x, subnet_vars):
		y1 = tf.nn.sigmoid(batch_norm(conv2d(x, subnet_vars[0], 1), FLAGS.train_mode))
		y2 = tf.nn.sigmoid(batch_norm(conv2d(x, subnet_vars[2], 1), FLAGS.train_mode))
		y2 = tf.nn.sigmoid(batch_norm(conv2d(y2, subnet_vars[4], 1), FLAGS.train_mode))
		y3 = tf.nn.sigmoid(batch_norm(conv2d(x, subnet_vars[6], 1), FLAGS.train_mode))
		y3 = tf.nn.sigmoid(batch_norm(conv2d(y3, subnet_vars[8], 1), FLAGS.train_mode))
		y4 = max_pool_3x3(x, 1)
		y4 = tf.nn.sigmoid(batch_norm(conv2d(y4, subnet_vars[10], 1), FLAGS.train_mode))
		y = tf.concat([y1, y2, y3, y4], 3)
		return y
		
	def network(self, x):
		if FLAGS.using_batch_norm:
			shape = tf.shape(x)
			width = tf.cast(shape[1], tf.float32)
			height = tf.cast(shape[2], tf.float32)
			y = conv2d(x, self.conv1, 1)
			y = tf.nn.sigmoid(batch_norm(y, FLAGS.train_mode))
			y = max_pool_3x3(y, 2)
			y = conv2d(y, self.conv2, 1)
			y = tf.nn.sigmoid(batch_norm(y, FLAGS.train_mode))
			y = max_pool_3x3(y, 2)
			y = self._subnetwork_bn(y, self.subnet_vars1)
			y = max_pool_3x3(y, 2)
			y = self._subnetwork_bn(y, self.subnet_vars2)
			y = self._subnetwork_bn(y, self.subnet_vars3)
			y = max_pool_3x3(y, 2)
			y = self._subnetwork_bn(y, self.subnet_vars4)
			y = self._subnetwork_bn(y, self.subnet_vars5)
			y = max_pool_3x3(y, 2)
			y = self._subnetwork_bn(y, self.subnet_vars6)
			y = self._subnetwork_bn(y, self.subnet_vars7)
			y = max_pool_3x3(y, 2)
			y = self._subnetwork_bn(y, self.subnet_vars8)
			
			y_prob = tf.sigmoid(batch_norm(conv2d(y, self.conv_prob, 1), FLAGS.train_mode))			#shape=[-1, 6, 10, 5]
			y_center = tf.sigmoid(batch_norm(conv2d(y, self.conv_center, 1), FLAGS.train_mode))	#shape=[-1, 6, 10, 10]
			y_size = tf.exp(conv2d(y, self.conv_size, 1) + self.bias_size)				#shape=[-1, 6, 10, 10]
		else:
			shape = tf.shape(x)
			width = tf.cast(shape[1], tf.float32)
			height = tf.cast(shape[2], tf.float32)
			y = conv2d(x, self.conv1, 1) + self.bias1
			y = leaky(y)
			y = max_pool_3x3(y, 2)
			y = conv2d(y, self.conv2, 1) + self.bias2
			y = leaky(y)
			y = max_pool_3x3(y, 2)
			y = self._subnetwork(y, self.subnet_vars1)
			y = max_pool_3x3(y, 2)
			y = self._subnetwork(y, self.subnet_vars2)
			y = self._subnetwork(y, self.subnet_vars3)
			y = max_pool_3x3(y, 2)
			y = self._subnetwork(y, self.subnet_vars4)
			y = self._subnetwork(y, self.subnet_vars5)
			y = max_pool_3x3(y, 2)
			y = self._subnetwork(y, self.subnet_vars6)
			y = self._subnetwork(y, self.subnet_vars7)
			y = max_pool_3x3(y, 2)
			y = self._subnetwork(y, self.subnet_vars8)
			
			y_prob = tf.sigmoid(conv2d(y, self.conv_prob, 1) + self.bias_prob)			#shape=[-1, 6, 10, 5]
			y_center = tf.sigmoid(conv2d(y, self.conv_center, 1) + self.bias_center)	#shape=[-1, 6, 10, 10]
			y_size = tf.exp(conv2d(y, self.conv_size, 1) + self.bias_size)				#shape=[-1, 6, 10, 10]
		
		#y_center = y_center*self.cell_size + self.cell_left_top
		y_center = tf.reshape(y_center, [-1, self.cell_shape[0], self.cell_shape[1], 5, 2])
		y_center = tf.multiply(y_center, self.cell_size)
		y_center = tf.transpose(y_center, perm=[0, 3, 1, 2, 4])
		y_center = tf.add(y_center, self.cell_left_top)
		y_center = tf.transpose(y_center, [0, 2, 3, 1, 4])		#shape=[-1, 6, 10, 5, 2]
		#y_size = y_size*self.anchor_box_size
		y_size = tf.reshape(y_size, [-1, FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width, 5, 2])
		y_size = tf.multiply(y_size, self.anchor_box_size)		#shape=[-1, 6, 10, 5, 2]
		y_coor = y_center - y_size/2
		
		ret_val = [y_prob, y_coor, y_size]
		
		return ret_val
		
	def loss_func(self, y, y_):	#y[0]:probability, y[1]:x, y[2]:y, y[3]:width, y[4]:height
		prob_loss = tf.reduce_mean(-(FLAGS.valid_box_prob_weight*y[0]*tf.log(y_[0]+1E-10)+(1-y[0])*tf.log(1-y_[0]+1E-10)))
		y_coor_x, y_coor_y = tf.split(y[1], 2, 4)
		y__coor_x, y__coor_y = tf.split(y_[1], 2, 4)
		y_size_width, y_size_height = tf.split(y[2], 2, 4)
		y__size_width, y__size_height = tf.split(y_[2], 2, 4)
		y_coor_x = tf.squeeze(y_coor_x, [-1])
		y__coor_x = tf.squeeze(y__coor_x, [-1])
		y_coor_y = tf.squeeze(y_coor_y, [-1])
		y__coor_y = tf.squeeze(y__coor_y, [-1])
		y_size_width = tf.squeeze(y_size_width, [-1])
		y__size_width = tf.squeeze(y__size_width, [-1])
		y_size_height = tf.squeeze(y_size_height, [-1])
		y__size_height = tf.squeeze(y__size_height, [-1])
		coor_loss = tf.reduce_mean((tf.square((y_coor_x-y__coor_x)/(y_size_width+1E-10)) + tf.square((y_coor_y-y__coor_y)/(y_size_height+1E-10)))*y[0])
		size_loss = tf.reduce_mean((tf.square(y_size_width/(y__size_width+1E-10)-1) + tf.square(y_size_height/(y__size_height+1E-10)-1))*y[0])
		loss = FLAGS.prob_loss_weight*prob_loss + FLAGS.coor_loss_weight*coor_loss + FLAGS.size_loss_weight*size_loss
		return loss
		
	def load_check_point(self, session):
		ckpt = tf.train.get_checkpoint_state(FLAGS.check_point_dir)
		ckpt_name = FLAGS.check_point_load_file
		if ckpt_name:
			self.saver.restore(session, os.path.join(FLAGS.check_point_dir, ckpt_name))
		else:
			if ckpt and ckpt.model_checkpoint_path:
				ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
				self.saver.restore(session, os.path.join(FLAGS.check_point_dir, ckpt_name))
		return ckpt_name
			
	def save_check_point(self, session, step):
		self.saver.save(session, os.path.join(FLAGS.check_point_dir, "pd.model"), global_step=step)
		
	def print_variables(self, session, x_placeholder=None, y_placeholder=None):
		batch_size = FLAGS.train_batch_size
		name_list = get_name_list(FLAGS.train_img_dir)
		name_amt = len(name_list)
		input_names = get_next_batch(name_list, (0*batch_size)%name_amt, batch_size)
		input_x, input_y = get_input_and_ground_truth_data(input_names)
		y_center = session.run(self.y_center, feed_dict={x_placeholder:input_x, y_placeholder:input_y})

		
def get_anchor_box_size():
	return ((20, 40), (36, 72), (62, 124), (106, 212), (180, 360))
	
def get_cell_size():
	return [FLAGS.input_img_width/FLAGS.output_cell_shape_width, FLAGS.input_img_height/FLAGS.output_cell_shape_height]
	
def get_output_cell_left_top():
	left_top = []
	cell_size = get_cell_size()
	for i in range(FLAGS.output_cell_shape_height):
		left_top += [[]]
		for j in range(FLAGS.output_cell_shape_width):
			left_top[i] += [[j*cell_size[0], i*cell_size[1]]]
	return left_top

def iou(box1, box2):
	if box1[0] >= box2[0]+box2[2]: return 0
	if box2[0] >= box1[0]+box1[2]: return 0
	if box1[1] >= box2[1]+box2[3]: return 0
	if box2[1] >= box1[1]+box1[3]: return 0
	iou_width = min(box1[0]+box1[2], box2[0]+box2[2]) - max(box1[0], box2[0])
	iou_height = min(box1[1]+box1[3], box2[1]+box2[3]) - max(box1[1], box2[1])
	iou_area = iou_width*iou_height
	box1_area = box1[2]*box1[3]
	box2_area = box2[2]*box2[3]
	return iou_area/(box1_area+box2_area-iou_area)	
		
def get_name_list(dir):
	name_list = []
	for i, file in enumerate(os.listdir(dir)):
		if os.path.isfile(os.path.join(dir, file)):
			name_list += [os.path.splitext(file)[0]]
	random.shuffle(name_list)
	return name_list
	
def get_next_batch(name_list, offset, batch_size):
	if len(name_list) <= (offset+batch_size):
		random.shuffle(name_list)
		print("shuffle")
		files = name_list[:batch_size]
	else:
		files = name_list[offset:offset+batch_size]
		
	return files
	
def get_resize_size(src_width, src_height):
	if src_width/src_height > FLAGS.input_img_width/FLAGS.input_img_height:
		res_img_width = FLAGS.input_img_width
		res_img_height = src_height*FLAGS.input_img_width/src_width
	else:
		res_img_width = src_width*FLAGS.input_img_height/src_height
		res_img_height = FLAGS.input_img_height
	return res_img_width, res_img_height
		
def get_input_data(names):
	data = []
	for i in names:
		if FLAGS.train_mode:
			file = os.path.join(FLAGS.train_img_dir, i+FLAGS.image_file_extension)
		else:
			file = os.path.join(FLAGS.test_img_dir, i+FLAGS.image_file_extension)
		src_img = Image.open(i)
		dst_img = Image.new('RGB', (FLAGS.input_img_width, FLAGS.input_img_height))
		res_img_width, res_img_height = get_resize_size(src_img.size[0], src_img.size[1])
		res_img = src_img.resize((int(res_img_width), int(res_img_height)))
		dst_img.paste(res_img, (0, 0))
		data += [np.array(dst_img)]
	return data
	
def get_groundtruth_by_size(object_info):
	box_sizes = get_anchor_box_size();
	cell_left_top = get_output_cell_left_top()
	cell_size = get_cell_size()
	cell_row = 0
	cell_col = 0
	gt_prob = np.zeros([FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width, FLAGS.anchor_box_amt])
	gt_coor = np.zeros([FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width, FLAGS.anchor_box_amt, 2])
	gt_size = np.zeros([FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width, FLAGS.anchor_box_amt, 2])
	box_area_sqrt = []
	for box_size in box_sizes:
		box_area_sqrt += [math.sqrt(box_size[0]*box_size[1])]
	for object in object_info:
		obj_area_sqrt = math.sqrt(object[2]*object[3])
		bigger_cell_pos = FLAGS.anchor_box_amt
		object_center = [object[0]+object[2]/2, object[1]+object[3]/2]
		cell_row = (int)(object_center[1]//cell_size[1])
		cell_col = (int)(object_center[0]//cell_size[0])
		for i, cur_box_area_sqrt in enumerate(box_area_sqrt):
			if cur_box_area_sqrt > obj_area_sqrt:
				bigger_cell_pos = i
				break
		for i in [-1, 0]:
			pos = bigger_cell_pos+i
			if pos>=0 and pos<FLAGS.anchor_box_amt:
				gt_prob[cell_row][cell_col][pos] = 1
				gt_coor[cell_row][cell_col][pos] = [object[0], object[1]]
				gt_size[cell_row][cell_col][pos] = [object[2], object[3]]
			
	return [gt_prob, gt_coor, gt_size]
	
def get_groundtruth_by_iou(object_info):
	box_sizes = get_anchor_box_size();
	cell_left_top = get_output_cell_left_top()
	cell_size = get_cell_size()
	cell_row = 0
	cell_col = 0
	gt_prob = np.zeros([FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width, FLAGS.anchor_box_amt])
	gt_coor = np.zeros([FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width, FLAGS.anchor_box_amt, 2])
	gt_size = np.zeros([FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width, FLAGS.anchor_box_amt, 2])
	for object in object_info:
		valid_box_cnt = 0
		max_iou_val = 0
		max_iou_cell_pos = 0
		object_center = [object[0]+object[2]/2, object[1]+object[3]/2]
		cell_row = (int)(object_center[1]//cell_size[1])
		cell_col = (int)(object_center[0]//cell_size[0])
		cell_center = [cell_col*cell_size[0]+cell_size[0]/2, cell_row*cell_size[1]+cell_size[1]/2]
		for i, box_size in enumerate(box_sizes):
			box = [cell_center[0]-box_size[0]/2, cell_center[1]-box_size[1]/2, box_size[0], box_size[1]]
			iou_val = iou(object, box)
			if iou_val > max_iou_val:
				max_iou_val = iou_val
				max_iou_cell_pos = i
			if FLAGS.iou_thresh<iou_val and 0==gt_prob[cell_row][cell_col][i]:
				gt_prob[cell_row][cell_col][i] = 1
				gt_coor[cell_row][cell_col][i] = [object[0], object[1]]
				gt_size[cell_row][cell_col][i] = [object[2], object[3]]
				valid_box_cnt = 1
		if 0==valid_box_cnt and 0==gt_prob[cell_row][cell_col][max_iou_cell_pos]:
			gt_prob[cell_row][cell_col][max_iou_cell_pos] = 1
			gt_coor[cell_row][cell_col][max_iou_cell_pos] = [object[0], object[1]]
			gt_size[cell_row][cell_col][max_iou_cell_pos] = [object[2], object[3]]
			
	return [gt_prob, gt_coor, gt_size]
			
def get_ground_truth_data(shapes, names):
	data = []
	for i in names:
		if FLAGS.train_mode:
			file = os.path.join(FLAGS.train_ground_truth_dir_dir, i+FLAGS.ground_truth_file_extension)
		else:
			file = os.path.join(FLAGS.test_ground_truth_dir, i+FLAGS.ground_truth_file_extension)
		with open(file, "r") as file_handle:
			line = file_handle.readline()
			
def get_input_and_ground_truth_data(names):
	x_data = []
	y_data = []
	y_prob = []
	y_coor = []
	y_size = []
	
	for i,name in enumerate(names):
		if FLAGS.train_mode:
			file = os.path.join(FLAGS.train_img_dir, name+FLAGS.image_file_extension)
		else:
			file = os.path.join(FLAGS.test_img_dir, name+FLAGS.image_file_extension)
		src_img = Image.open(file)
		dst_img = Image.new('RGB', (FLAGS.input_img_width, FLAGS.input_img_height))
		res_img_width, res_img_height = get_resize_size(src_img.size[0], src_img.size[1])
		res_img = src_img.resize((int(res_img_width), int(res_img_height)))
		dst_img.paste(res_img, (0, 0))
		x_data += [np.array(dst_img)]
		
		res_to_src_size_ratio = [res_img_width/src_img.size[0], res_img_height/src_img.size[1]]
		if FLAGS.train_mode:
			file = os.path.join(FLAGS.train_ground_truth_dir, name+FLAGS.ground_truth_file_extension)
		else:
			file = os.path.join(FLAGS.test_ground_truth_dir, name+FLAGS.ground_truth_file_extension)
		with open(file, "r") as file_handle:
			object_info = []
			for line in file_handle:
				x, y, width, height = line.split(" ")
				size = [float(width)*res_to_src_size_ratio[0], float(height)*res_to_src_size_ratio[1]]
				coor = [float(x)*res_to_src_size_ratio[0], float(y)*res_to_src_size_ratio[1]]
				object_info += [coor+size]
			y = get_groundtruth_by_size(object_info)
			y_prob += [y[0]]
			y_coor += [y[1]]
			y_size += [y[2]]
				
	y_data = [y_prob, y_coor, y_size]			
	
	return x_data, y_data
	
def load_check_point(session, network):
	ckpt_name = network.load_check_point(session)
	if 0 != len(ckpt_name):
		print("Success to load {}".format(ckpt_name))
	else:
		print("Failed to find a checkpoint")
		
def print_contrast(input_y, y_val):
	#prob
	'''for i in range(len(input_y[0])):
		print("gt prob:", input_y[0][i])
		print("res prob:", y_val[0][i])
		print("")'''
	for i in range(len(input_y[0])):
		for row in range(len(input_y[0][i])):
			for col in range(len(input_y[0][i][row])):
				for j in range(len(input_y[0][i][row][col])):
					if 0 != input_y[0][i][row][col][j]:
						print("gt:", input_y[0][i][row][col][j], input_y[1][i][row][col][j], input_y[2][i][row][col][j])
						print("res:", y_val[0][i][row][col][j], y_val[1][i][row][col][j], y_val[2][i][row][col][j])
						print("")
					
def train_network():
	x = tf.placeholder("float", [None, FLAGS.input_img_height, FLAGS.input_img_width, 3])
	y_prob = tf.placeholder("float", [None, FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width, FLAGS.anchor_box_amt])
	y_coor = tf.placeholder("float", [None, FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width, FLAGS.anchor_box_amt, 2])
	y_size = tf.placeholder("float", [None, FLAGS.output_cell_shape_height, FLAGS.output_cell_shape_width, FLAGS.anchor_box_amt, 2])
	pd_network = PD_network()

	batch_size = FLAGS.train_batch_size

	name_list = get_name_list(FLAGS.train_img_dir)
	name_amt = len(name_list)
	
	y_ = pd_network.network(x)
	y = [y_prob, y_coor, y_size]
	loss = pd_network.loss_func(y, y_)
	tf.summary.scalar("loss", loss)
	trainer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
	#trainer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
	init = tf.global_variables_initializer()
	
	with tf.Session() as session:
		merged = tf.summary.merge_all()
		for file in os.listdir(FLAGS.graph_dir): os.remove(os.path.join(FLAGS.graph_dir, file))
		tb_writer = tf.summary.FileWriter(FLAGS.graph_dir, session.graph)
		session.run(init)
		load_check_point(session, pd_network)
		for step in range(FLAGS.max_learning_step):
			input_names = get_next_batch(name_list, (step*batch_size)%name_amt, batch_size)
			input_x, input_y = get_input_and_ground_truth_data(input_names)
			_, tb_info, loss_val, y_val = session.run([trainer, merged, loss, y_], feed_dict={x:input_x, y_prob:input_y[0], y_coor:input_y[1], y_size:input_y[2]})
			tb_writer.add_summary(tb_info, step)
			print_contrast(input_y, y_val)
			print(step, loss_val)
			print("")
			if 0==step%FLAGS.session_save_step and 0!=step:
				pd_network.save_check_point(session, step)
				print("check point saved")
		tb_writer.close()

def test_network():
	x = tf.placeholder("float", [None, FLAGS.input_img_height, FLAGS.input_img_width, 3])
	pd_network = PD_network()

	test_file_path = FLAGS.test_img_dir
	test_file_name = []
	files = os.listdir(test_file_path)
	files.sort()
	for file in files:
		file_path = os.path.join(test_file_path, file)
		if os.path.isfile(file_path):
			test_file_name += [file_path]
	print(test_file_name)
	y_ = pd_network.network(x)
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		load_check_point(session, pd_network)
		input_x = get_input_data(test_file_name)
		y_ = session.run(y_, feed_dict={x:input_x})
	
	img_num = 0
	for input in input_x:
		dst_img = Image.fromarray(input)
		draw = ImageDraw.Draw(dst_img)		
		for row in range(FLAGS.output_cell_shape_height):
			for col in range(FLAGS.output_cell_shape_width):
				for box_num in range(FLAGS.anchor_box_amt):
					if y_[0][img_num][row][col][box_num] > FLAGS.valid_prob_thresh:
						coor_x1 = int(y_[1][img_num][row][col][box_num][0])
						coor_y1 = int(y_[1][img_num][row][col][box_num][1])
						coor_x2 = coor_x1 + int(y_[2][img_num][row][col][box_num][0])
						coor_y2 = coor_y1 + int(y_[2][img_num][row][col][box_num][1])
						draw.rectangle([coor_x1, coor_y1, coor_x2, coor_y2], outline = 128)
		dst_img.save(os.path.join(FLAGS.test_img_dir, str(img_num)+".jpg"))
		img_num += 1
		del draw
		
		
if __name__ == '__main__':
	if FLAGS.train_mode:
		train_network()
	else:
		test_network()
	
