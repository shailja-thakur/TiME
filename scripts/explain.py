import numpy as np 
import sys
import os
from utils.utils import reshape, label_encoding, calculate_metrics, smooth, norm, torch_norm, plot_samples, plot_score_densities
import utils
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F

# generate weight vector for the class specific model attribution such that 
# input sample with maximum contribution has maximum weight and the input samples 
# with minimum contribution are assigned minimum weights

class Explain:


	def __init__(self, X_test, inp_shape, N, fig_dir, model, classes, window_name=utils.constants.window_name, 
							window_size=utils.constants.window_size, lobe=utils.constants.max_lobe,
							batch=utils.constants.batch,epochs=utils.constants.epoch):

		self.fig_dir = fig_dir
		self.model = model
		self.test = X_test
		self.N = N
		self.classes = classes
		self.inp_shape = inp_shape

		self.window_name = window_name
		
		self.window_size = window_size
		
		self.lobe=lobe

		self.epochs=epochs

		self.batch=batch

		self.lr = utils.constants.lr

				
		if len(inp_shape) == 2:

			self.ndim = inp_shape[0] if inp_shape[1]<inp_shape[0] else inp_shape[1]
	    
		print('Loaded configuration as follows ....')
		print('N:{}'.format(self.N))
		print('Classes:{}'.format(self.classes))
		print('Input shape:{}'.format(self.inp_shape))
		print('Window Name:{}'.format(self.window_name))
		print('Window Size:{}'.format(self.window_size))
		print('Maximum Lobes (rising/falling edges):{}'.format(self.lobe))
		print('Batch:{}'.format(batch))
		print('Epochs:{}'.format(epochs))
		print('Dimension:{}'.format(self.ndim))

		test_acc = 0.0

		# if not isinstance(self.test, np.ndarray):
		# 	for samples, labels in self.test:

		# 		with torch.no_grad():
		# 			samples, labels = samples.cuda(), labels.cuda()
		# 			output = self.model(samples)
		# 			# calculate accuracy
		# 			pred = torch.argmax(output, dim=1)
		# 			correct = pred.eq(labels)
		# 			test_acc += torch.mean(correct.float())

		# 	print('Accuracy of the network on {} test images: {}%'.format(N, round(test_acc.item()*100.0/len(self.test), 2)))



	def get_gaussian_noise_samples(self,p=0.5):
    
	    grids = np.empty((self.batch, self.ndim))
	    samples = np.random.normal(0,1, size=(self.batch, self.ndim)) < p
	    samples = samples.astype('float32')
	    for i in range(0, self.batch):
	        grids[i] = smooth(samples[i], 20)[0:self.ndim]
	    return grids


	def gen_random_binary_mask_samples(self):
	    
	    # print('(self.batch, # dim)', self.batch, self.ndim)
	    max_lobes =  self.lobe
	    high = int(self.ndim/(self.ndim/max_lobes))
	    win = int((self.window_size/100)*self.ndim)
	    
	    low = np.random.randint(1,high=high, size=self.batch)
	    # print(high, win)
	    rng = np.random.default_rng()
	    
	    samples = np.concatenate((np.ones((self.batch,low[0])),np.zeros((self.batch, self.ndim-low[0]))),axis=1)
	    for i,sample in enumerate(samples):
	    
	        rng.shuffle(sample)
	        
	        samples[i]=smooth(sample, win,window=self.window_name)[0:self.ndim]
	        
	    samples = samples.astype('float32')
	    
	    return samples

	

	def get_imp_weight_vector_physionet(self):

		

		# if int((6/10)*self.ndim) <= self.window_size: return
		# print(int((6/10)*self.ndim) , self.window_size)

		n_classes = len(self.classes)

		sal = np.ones((self.N, self.ndim, n_classes))      
		latency = np.empty((self.N, 1))
		total_preds = np.empty((self.N, self.epochs, self.batch, n_classes))
		total_true_preds = np.empty((self.N, self.epochs, n_classes))


		# print(self.inp_shape, len(self.test))
		
		test_acc = 0.0
		for x, row in enumerate(self.test):
			
			with torch.no_grad():

				sample, label = row
				sample, label = sample.cuda(), label.cuda()

				start = time.time()

				sal_itrs = np.empty((self.epochs, self.ndim, n_classes))   
				preds = np.empty((self.epochs, 1,n_classes))      
				temp_masked_preds = np.empty((self.epochs, self.batch, n_classes))
				temp_true_preds = np.empty((self.epochs, n_classes))
				# print(sal_itrs.shape, temp_masked_preds.shape)
				j=0
				for i in range(0, self.epochs):
					
					
					M=self.gen_random_binary_mask_samples()
					M_noise=self.get_gaussian_noise_samples()
					

					M = M.reshape(self.batch, *self.inp_shape)
					M_noise = M_noise.reshape(self.batch, *self.inp_shape)

					M = norm(M)
					M = M*M_noise
					M = torch.from_numpy(M).type(torch.cuda.FloatTensor).cuda()
					# test_input = norm(samples)
					# print(sample.cpu().numpy())
					# print(M.shape, sample.shape)
					masked_inputs = torch_norm(M*sample)
					# masked_inputs = masked_inputs
					# print(masked_inputs.shape)

					# print(masked_inputs)


					# print(samples.shape, labels.shape)
					true_preds = self.model(sample)
					# print(true_preds.shape)
					# calculate accuracy					
					masked_preds = self.model(masked_inputs)
					unmasked_preds = self.model(1-masked_inputs)
					# print(masked_preds.cpu().numpy().reshape(self.batch, n_classes))

					# print(masked_preds.size(), M[:,0,:].size())
					M_grads_on= torch.mul(torch.matmul(masked_preds.T, M[:,0,:]),torch.tensor(self.lr).cuda())
					M_grads_anti= torch.mul(torch.matmul(unmasked_preds.T, M[:,0,:]),torch.tensor(self.lr).cuda())
					# print(M_grads_on.size())
					# true_preds = true_preds
					# masked_preds = masked_preds
					M_grads_on = M_grads_on.cpu().numpy()
					M_grads_anti = M_grads_anti.cpu().numpy()

					temp_on= M_grads_on
					temp_off = M_grads_anti
					temp_masked_preds[i]=masked_preds.cpu().numpy().reshape(self.batch, n_classes)
					temp_true_preds[i]= true_preds.cpu().numpy().reshape(n_classes)
					sal_itrs[i] = (temp_on.T)/self.batch    
					# print(sal_itrs.shape)
				

				temp = np.einsum('ijk->jk',sal_itrs)
				duration = time.time() - start
				# print(temp.shape)

				# print(np.mean(temp_masked_preds, axis=0).shape)

				# masked_preds_inputs[x]=np.mean(temp_masked_preds, axis=0)
				sal[x] = (temp/self.epochs)
				latency[x]=duration
				total_preds[x]=temp_masked_preds
				total_true_preds[x]=temp_true_preds

		print(sal.shape, total_preds.shape, latency.shape)

		return sal, total_preds, latency

		# print('Accuracy of the network on {} test images: {}%'.format(self.N, round(test_acc.item()*100.0/len(self.test), 2)))

	    
	def get_imp_weight_vector(self):

		# if int((6/10)*self.ndim) <= self.window_size: 
		# if int(self.ndim) <= self.window_size: 
			# print('window size greater than 60 percent of the length of the input segment')
			# return

		n_classes = len(self.classes)

		sal = np.ones((self.N, self.test.shape[1], n_classes))      
		latency = np.empty((self.N, 1))
		total_preds = np.empty((self.N, self.epochs, self.batch, n_classes))
		# total_true_preds = np.empty((self.N, self.epochs, n_classes))
		for x,test_input in enumerate(self.test):
			start = time.time()

			sal_itrs = np.empty((self.epochs, self.test.shape[1], n_classes))   
			preds = np.empty((self.epochs, 1,n_classes))      
			temp_masked_preds = np.empty((self.epochs, self.batch, n_classes))
			# temp_true_preds = np.empty((self.epochs, n_classes))

			j=0
			for i in range(0, self.epochs):
				M=self.gen_random_binary_mask_samples()
				M_noise=self.get_gaussian_noise_samples()
				
				M = M.reshape(M.shape[0], M.shape[1],1)
				M_noise = M_noise.reshape(M_noise.shape[0], M_noise.shape[1],1)

				
				M = norm(M)
				M = M*M_noise
				test_input = norm(test_input)
				
				masked_inputs = norm(M*test_input)
				true_preds = self.model.predict(test_input.reshape(1,test_input.shape[0],1))
				masked_preds = self.model.predict(masked_inputs)
				unmasked_preds = self.model.predict(1-masked_inputs)
				true_class = np.argmax(true_preds)

				# print('fig path', self.fig_dir)
				# plot_score_densities(masked_preds, self.fig_dir, 'plot_score_densities')

				# print('pred class',true_class, 'conf scores',true_preds)
				# print('masked pred',np.argmax(masked_preds[2]))

				#         multiply the masks with confidence scores and sum over the masks to get a final saliency 
				M_grads_on= (masked_preds).T.dot(((M[:,:,0])))*self.lr  
				M_grads_anti= (unmasked_preds).T.dot(((M[:,:,0])))*(self.lr)

				temp_on=((M_grads_on))
				temp_off = ((M_grads_anti))

				temp_masked_preds[i]=masked_preds
				# temp_true_preds[i]=true_preds
				sal_itrs[i] = (((temp_on)).T)/self.batch      

				# filename='ON gradient with confidence score, Predicted Class = {},iteration {}'.format(np.argmax(true_preds[0]), i)
				# plot_samples(test_input, temp_on, self.fig_dir, filename, self.classes,save=True, title=False, plot=False)

				# filename='OFF, Predicted Class = {},iteration {}'.format(np.argmax(true_preds[0]), i)
				# plot_samples(test_input, temp_off, self.fig_dir, filename, self.classes, save=True, title=False, plot=False)
				

			temp = np.einsum('ijk->jk',sal_itrs)
			duration = time.time() - start

			# print(np.mean(temp_masked_preds, axis=0).shape)

			# masked_preds_inputs[x]=np.mean(temp_masked_preds, axis=0)
			sal[x] = (temp/self.epochs)
			latency[x]=duration
			total_preds[x]=temp_masked_preds
			# total_true_preds[x]=temp_true_preds

			# filename='Average Saliency, Predicted Class = {}'.format(np.argmax(true_preds[0]))
			# plot_samples(test_input, sal[x].T, self.fig_dir, filename, self.classes, save=True, title=False, plot=False)

		print(sal.shape, total_preds.shape, latency.shape)

		return sal, total_preds, latency




