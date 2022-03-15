# THE PROJECT MINI!!

# Questions

# can I do percentages instead of set values for S?
# are the plots I have what you're looking for?
# do we need to put our implementation inside of imgRecover in class_utils?

# Notes
# 	Instead of plotting mean lambda and error, scatter them all
# 	lasso error might be calculated better in a different way (see book / line 352)

# To Do
	# Vary k, what happens?
	# estimate k automatically


from sys import argv
import os
import time
import random
import math
from multiprocessing import Pool

from class_utils import *

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate, RepeatedKFold

import numpy as np
from scipy.signal import medfilt2d

def deconstruct_image(img,height,width,k):

	"""
	Break image into kxk blocks

	return:
		img: a COPY of that image in the form of n x k**2 vectors (n=number of blocks)

	"""

	print("Deconstructing image into blocks")
	if (((height % k) != 0) or ((width % k) != 0) or (k == 0)):
		raise TypeError("Yo, hold on, you can't divide this image evenly into " + str(k) + "x" + str(k) + " blocks!")
	num_r = height // k # number of rows of blocks
	num_c = width // k # number of columns of blocks
	num_blocks = num_r * num_c
	blocks = np.zeros([num_blocks,k,k])

	# go through all blocks and save them!
	n = 0 #index for blocks
	for r in range(num_r): # assuming n>0
		x_offset = (r % num_r) * k
		for c in range(num_c):
			y_offset = (c % num_c) * k
			blocks[n,:,:] = img[x_offset:x_offset+k,y_offset:y_offset+k].copy()
			n+=1

	return blocks.reshape(num_blocks,k**2)


def corrupt_blocks(blocks,S,corruption_value=np.inf):
	"""
	Corrupt k^2 - S pixels in place!

	return:
		None.

	"""
	print("Corrupting blocks")
	k_sq = np.shape(blocks)[1]
	num_to_corrupt = k_sq - S

	for n in range(np.shape(blocks)[0]):
		current_block = blocks[n]
		pixel_idx = list(range(k_sq))
		rng = np.random.default_rng() # init prng
		rng.shuffle(pixel_idx)
		#np.random.shuffle(pixel_idx)
		for pixel in range(num_to_corrupt):
			current_block[pixel_idx[pixel]] = corruption_value

def compute_transformation_matrix(P,Q):
	"""
	Take an image and compute transformation matrix!

	:param P: # rows of image
	:param Q: # cols of image

	Why not use values of pixels?

	:return: T - a transformation matrix
	"""
	print("Computing transformation matrix")
	T = np.zeros([P*Q,P*Q])

	for u in range(P): # first create a set u,v pair
		if u == 1:
			a_u = math.sqrt(1/P)
		else:
			a_u = math.sqrt(2/P)
		for v in range(Q):
			if v == 1:
				b_v = math.sqrt(1/Q)
			else:
				b_v = math.sqrt(2/Q)

			# now go through all pixels
			for x in range(P):
				for y in range(Q):
					# and create a column vector (Td) for the transformation matrix (T)
					cos1 = math.cos((math.pi * (2*x - 1) * (u - 1)) / 2*P) #split the equation into parts bc it's easier
					cos2 = math.cos((math.pi * (2*y - 1) * (v - 1)) / 2*Q)
					T[x,y] = a_u*b_v*cos1*cos2

			# repeat for all u,v pairs

	return T


def lasso_regression_withcv(B,A,m,lambda_range=[-6,6],M=20):

	"""
	Run Lasso Regression with random subset cross validation

	Params
	------

	B : subset of image containing only S sensed pixels
	A : subset of k**2 x k**2 transf. matrix, containing only S rows corr. to S sensed pixels
	m : number of samples to take for testing (S - m for training)
	M : number of repetitions
	"""

	lambdas = np.logspace(lambda_range[0],lambda_range[1],num=13) # set up range of gammas in logscale
	# lambdas /= lambdas/(np.shape(B)[0]) # sklearn
	best_error = np.ones([1,m])*np.inf
	# np.zeros([np.shape(B)-m])
	optimal_lambda = 0
	best_coefs = np.ones([1,m])*np.inf
	#lx = 0
	for lambda_ in lambdas: # go through each candidate gamma
		#print("Testing lambda=" + str(lambda_) +" ("+str(lx+1)+" of " +str(len(lambdas))+ ") for "+str(M)+" iterations.")
		errors = [] # collect errors for each iteration of test/train subset split
		for Mx in range(M): # do this M times
			# randomly sample from 
			indices = np.arange(np.shape(B)[0]) # randomly select m indices
			rng = np.random.default_rng() # init prng
			rng.shuffle(indices)
			training_idx = indices[m:] # take last S-m samples for training
			testing_idx = indices[:m] # take first m samples for testing
			# train and test Lasso model
			lasso_model = Lasso(alpha=lambda_) # try max_iter=1000
			lasso_model.fit(A[training_idx,:],B[training_idx,:]) # fit the train set
			B_hat = np.atleast_2d(lasso_model.predict(A[testing_idx,:])) # predict the test set
			sq_err = (B[testing_idx,:].T - B_hat)**2 # squared error
			errors.append(sq_err) # get error for this iteration
		avg_error = np.mean(errors,axis=0) # now avg the errors after running M=20 times
		# import pdb; pdb.set_trace()
		#lx +=1
		if (avg_error < best_error).all(): # selection criteria 
			#print(lasso_model.coef_)
			optimal_lambda = lambda_
			best_coefs = lasso_model.coef_
			best_error = avg_error
	
	return best_coefs,optimal_lambda,best_error,lasso_model.intercept_ # return lambda of that model


def recover_block(block,S,T,corruption_value=np.inf,lambda_range=[-6,6],M=20):
	sensed_pixel_indices = np.where(block!=corruption_value)[0]
	corrupt_pixel_indices = np.where(block==corruption_value)[0]
	A = T[sensed_pixel_indices,:]
	m = S // 6 # floor(S/6) S=#sensedpixels
	B = np.atleast_2d(block[sensed_pixel_indices]).T # make column vector
	dct_weights,optimal_lambda,best_error,intercept = lasso_regression_withcv(B,A,m,lambda_range) # DCT coeff
	# import pdb; pdb.set_trace()
	# get full vector of estimated pixel values
	recovered_block = np.matmul(T,dct_weights)
	#print(recovered_block[corrupt_pixel_indices]+intercept)
	#import pdb;pdb.set_trace()
	block[corrupt_pixel_indices] = recovered_block[corrupt_pixel_indices]+intercept # replace corrupt values with new pixel values
	
	return optimal_lambda,best_error

def recover_image_from_blocks(img,S,T,corruption_value=np.inf,lambda_range=[-6,6],M=20):
	"""
	Take an image (deconstructed into kxk blocks) and recover its pixels by kxk blocks!

	each block is a k**2-long array of pixel values (some values corrupted with non-RGB corruption_value)

	return mean lambda across all blocks

	"""

	print("Recovering blocks w/ Xtra Lasso Regressiveness")
	print("--------------------------------------------------")
	
	start = time.time() # time this

	# go through and find the missing pixel indices and take out the corresponding
	# basis functions (row) in T

	#lambdas = [] # collect lambdas for each block
	#errors = []
	#n=0

	#with Pool() as p: # parallelize block recovery
		#iterable = [(block,S,T,corruption_value,lambda_range,M) for block in img]
		#	lambda_,error = p.map(recover_block,iterable)
		#results = [p.apply(recover_block,args=(block,S,T,corruption_value,lambda_range,M)) for block in img]
		#import pdb; pdb.set_trace()
		#print(os.cpu_count())
		#lambdas.append(results[0])
		#errors.append(results[1])

	for block in img:
		recover_block(block,S,T,corruption_value,lambda_range,M)
		
	end = time.time()
	time_str = '{0:.2f}'.format(end-start)
	print("************************************************************")
	print("All blocks recovered! Total runtime of operation: "+time_str+" seconds.")
	print("************************************************************")
	#import pdb; pdb.set_trace()
	# return np.mean(lambdas),np.mean(errors) # get mean of lambdas to plot later!
	return

def stitch_blocks(blocks,height,width,k):

	print("Stitching blocks back together")
	img = np.zeros([height,width])

	num_c = width // k
	num_r = height // k
	num_blocks = num_r * num_c

	# go through all blocks and save them!
	n = 0 #index for blocks
	for r in range(num_r): # assuming n>0
		x_offset = (r % num_r) * k
		for c in range(num_c):
			y_offset = (c % num_c) * k
			block = blocks[n].copy().reshape(k,k)
			img[x_offset:x_offset+k,y_offset:y_offset+k] = block
			n+=1

	return img

def median_filter(img):

	print("Applying median filter")

	return medfilt2d(img)

def estimate_k(height,width):

	"""
	Take an image and estimate a k from its size
	"""
	#gcd = math.gcd(height,width)
	#print(height +"height")
	#k = np.max([height // gcd,  width // gcd])
	#print("GCD estimate of "+str(height)+"and "+str(width) +" is "+str(gcd))
	#print("Found k: " + str(k))
	return math.gcd(height,width)


def getRecoveryQuality(original,recovered,H,W):

	print("Calculating quality of image recovery")

	sum_ = 0
	for x in range(H):
		for y in range(W):
			sum_ += ((recovered[x,y] - original[x,y])**2)
	mse = sum_ * (1 / (W*H))
	return mse

def main(filename,k,subfolder):

	if subfolder[0] != '/':
		raise ValueError("Subdir folder must start with '/'")

	print("Entered main function....")
	print("\n*************************")
	
	img_original = imgRead(filename) # load image
	savedir = os.getcwd() + subfolder
	imgSave(savedir,filename,'unbothered',img=img_original)
	height = np.shape(img_original)[0]
	width = np.shape(img_original)[1]
	#print("Height: " + str(height))
	#print("Width: " + str(width))
	# k = estimate_k(height,width)
	#imgShow(img_original,title="Original Image") # show us!
	#percentages = [0.2,0.4,0.6,0.7,0.8,0.9] # do this recovery in a range of percentage of total pixels to corrupt, since k may change
	if k==8:
		Ss = [10,20,30,40,50]
		#Ss = [30]
	else:
		Ss = [10,30,50,100,150]
	#percentages = [0.7]
	filtered_RQs = []
	unfiltered_RQs = []
	avg_lambdas = []
	avg_errors = []
	for S in Ss:
		# display percentage complete
		#S = int(k*k * percent_sensed)
		print("\n\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
		img_copy = img_original.copy() # corrupt a copy of image
		img_copy.setflags(write=1) # make writeable
		img_blocks = deconstruct_image(img_copy,height,width,k)
		corrupt_blocks(img_blocks,S,corruption_value=np.inf) # scramble with np.inf so we can see
		img_corrupt = stitch_blocks(img_blocks,height,width,k)
		#imgShow(img_corrupt,title="Corrupted Image")
		imgSave(savedir,filename,'corrupted_S={S}'.format(S=S),img=img_corrupt)
		T = compute_transformation_matrix(k,k) # get T for all blocks
		lambda_range = [-6,6] # range of lambdas to test =[x,y], (1e^x,1e^y)
		recover_image_from_blocks(img_blocks,S,T,corruption_value=np.inf,lambda_range=lambda_range,M=20) # image recovery in place
		#avg_lambdas.append(lambda_) # save for later
		#avg_errors.append(error) # save for later
		
		# Record recovery qualities for each S (RQs) and save image before and after median filter
		recovered_img = stitch_blocks(img_blocks,height,width,k)
		unfiltered_RQs.append(getRecoveryQuality(img_original,recovered_img,height,width)) # plot later
		imgSave(savedir,filename,'recovered_S={S}'.format(S=S),img=recovered_img,cmap="gray",vmin=0,vmax=255)
		filtered_img = median_filter(recovered_img)
		filtered_RQs.append(getRecoveryQuality(img_original,filtered_img,height,width)) # plot later
		imgSave(savedir,filename,'recoveredfiltered_S={S}'.format(S=S),img=filtered_img,cmap="gray",vmin=0,vmax=255)

		total_percent_complete = '{0:.2f}'.format(((Ss.index(S)+1)/len(Ss))*100)

		print('Finished S={S}'.format(S=S) +"....... set is " + total_percent_complete + "% complete!")

	# Save results!
	print("Saving metadata from image recovery...")

	plt.figure(1)
	plt.plot(Ss,filtered_RQs)
	plt.title("Effect of Pixels Sensed (of total " + str(k*k) + " per block) on Recovery Quality (RQ)")
	plt.xlabel('Number of Pixels Sensed (S)')
	plt.ylabel('Mean-Squared Error (MSE)')
	index = filename.find('.')
	plt.savefig(savedir + '/' +filename[:index] + '_qualityresults' + '.png') # save plot

	plt.figure(2)
	plt.plot(Ss,unfiltered_RQs,color='magenta')
	plt.plot(Ss,filtered_RQs,color='blue')
	plt.legend(['Unfiltered Image','Filtered Image'])
	plt.title("Pre-Filter vs. Filtered RQ ")
	plt.xlabel('Number of Pixels Sensed (S)')
	plt.ylabel('MSE')
	plt.savefig(savedir + '/' +filename[:index] + '_filtercomparison' + '.png') # save plot

	"""

	plt.figure(3)
	plt.scatter(Sx,avg_lambdas)
	plt.xlim((lambda_range[0],lambda_range[1]))
	plt.title("Chosen Regularization Param per S")
	plt.xlabel('Number of Pixels Sensed (S)')
	plt.ylabel('Lambda')
	plt.savefig(savedir + '/' +filename[:index] + '_lambdascatter' + '.png') # save plot

	plt.figure(4)
	plt.scatter(Sx,avg_errors)
	plt.title("Avg. Lasso Classification Error per S") #lasso error might be calculated better in a different way (see book / line 351)
	plt.xlabel('Number of Pixels Sensed (S)')
	plt.ylabel('Error')
	plt.savefig(savedir + '/' +filename[:index] + '_lassoerrorscatter' + '.png') # save plot
	"""

if __name__ == "__main__":

	# usage: python3 [filename].png [k] /[subdir name]
	# k is block size

	# example: python3 mp1.py fishing_boat.bmp 8 /fishing_boat
	start = time.time()
	try:
		main(argv[1],int(argv[2]),argv[3])
	except KeyboardInterrupt:
		print(".\n.\n.\n.\n")
		raise KeyboardInterrupt("\n*******\n*******\n*******\nExecution suspended!")
	except RuntimeError:
		print(".\n.\n.\n.\n")
		raise RuntimeError("\n*******\n*******\n*******\nCorrect usage: python3 [image filename] [k] /[subdir name]" +
		".\n.\n.\n.\n.\n.\n.\n......(I'm not sure if this is actually a RuntimeError, I just wanted to use one :D)")
	
	# time execution
	print("Donezo.")
	#print("Total execution time was "+total_mins + " minutes" + " and "+total_secs+" seconds.")
