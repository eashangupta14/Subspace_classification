# Utility class 

import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2
from multiprocessing import Pool
import time
import concurrent.futures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import pandas as pd
import random
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


class integrate():
	"""This class applies rk integration method"""
	def __init__(self,y_init, x_input,step, b = 1,p =1,r=1,d=1):
		self.x_input = x_input
		self.y_init = y_init
		self.x_init = x_input[0]
		self.step = step
		self.results = {}
		self.b, self.p, self.r, self.d = b,p,r,d 
	
	def diff(self,x,y):
		# calculates derivative at x and y

		d = [(self.b - (self.p*y[1]))*y[0], ((self.r*y[0]) - self.d)*y[1]]
		d = np.array(d).reshape((len(d),1))
		return d


	def runge_kutta(self):
		""" This Solves the differential equation by using:

			y_n = y_(n-1) + k1/6 + k2/3 + k3/3 + k4/6

		"""
		y_rk = np.zeros((self.x_input.shape[0],self.y_init.shape[0],1), dtype = object)
		for pos,i in enumerate(self.x_input):
			if pos == 0:
				y_rk[pos,:,:] = self.y_init
			else:
				x,y = self.x_input[pos-1],y_rk[pos-1]
				k1 = self.step*self.diff(x,y)
				k2 = self.step*self.diff(x+(self.step/2),y+(k1/2))
				k3 = self.step*self.diff(x+(self.step/2),y+(k2/2))
				k4 = self.step*self.diff(x+self.step,y+k3)
				y_rk[pos,:,:] = y_rk[pos-1] + (k1/6) + (k2/3) + (k3/3) + (k4/6)   
		return y_rk
	
	def integrate(self):
		self.results = self.runge_kutta() 
		
	def plot(self):
		for i in range(self.y_init.shape[0]):
			plt.plot(self.x_input, self.results[:,i,0], 
					     label = f'x_{i+1}', marker='o')
		plt.xlabel('Time')
		plt.legend() 
		plt.show() 


class Classifier():
	"""docstring for Classifier"""
	def __init__(self, file_loc):
		self.base = file_loc
		self.train_tar = 0
		self.test_tar = 0
		self.train_dat = 0
		self.test_dat = 0
		self.classes = 0
		self.size = [32,32]
		self.gray = True
		self.pca = None
		self.pca_do = True
		self.lda = False
		self.lda_object = None
		self.train_folder = './temp/train'
		self.test_folder = './temp/test'
		self.model_folder = './temp/model'
		self.pred_folder = './temp/predictions'
	
	def process_data(self,f,image, gray = True,size = [64,64], test = False):
		img_loc = f'{f}/{image}'
		#print(img_loc)
		if gray:
			img = cv2.imread(img_loc,0)
			#img = cv2.GaussianBlur(img, (3,3), 0)
			img = cv2.resize(img, (size[0], size[1]))
			if test:
				return(cv2.normalize(img, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)), image
			else:
				return(cv2.normalize(img, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F))
		
	def load_test_mp(self, size = [64,64], gray = True, num_processes = 4):
		file_loc = f'{self.base}/Test'
		label_loc = f'{self.base}/Test.csv'
		df = pd.read_csv(label_loc)
		i = 0
		for image in os.listdir(file_loc):
			i = i+1

		test_data = np.zeros((i,size[0],size[1]))
		test_label = np.zeros(i)
		results = []
		#futures2 = []
		#with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
			# Submit tasks to the executor
		with Pool(num_processes) as pool:
			# Use the map function to apply the process_data function to each item in the loop
			results = pool.starmap(self.process_data, [(file_loc, image,gray,size,True) for image in os.listdir(file_loc)])
		#futures2 = list(tqdm(executor.submit(self.process_data,file_loc, image,gray,size) for image in os.listdir(file_loc)))
		#results2 = list(tqdm(future1.result() for future1 in futures2)) 
		results = np.array(results,dtype = object)
		test_data[:,:,:] = np.stack(results[:,0])
		test_label_mid = np.reshape(results[:,1],(results[:,1].shape[0],1))
		test_label_dict = dict(zip(df['Path'],df['ClassId']))
		
		for i,label in enumerate(test_label_mid):
			label = 'Test/'+ label[0].split('.png')[0].split('_')[0] + '.png'
			#print(type(test_label_dict.get(label,1000)))
			test_label[i] = test_label_dict.get(label,1000)
		self.test_tar = test_label
		data = np.reshape(test_data,(test_data.shape[0],test_data.shape[1]*test_data.shape[2]))
		self.test_dat = data
		#print('Data Loaded')
		if self.pca_do:
			
			print('Starting PCA on test dataset')
			self.test_dat = self.pca.transform(self.test_dat)
			print(self.test_dat.shape)
			print('Completed PCA on test dataset')

		if self.lda:
			print('Starting LDA on test Dataset')
			self.test_dat = self.lda_object.transform(self.test_dat)
			print(self.test_dat.shape)
			print('Completed LDA on test dataset')
	
	def load_train_mp2(self, size = [64,64], gray = True, num_processes = 4, num_components = 250, onlysvd = False):
		data = 0
		train_target = 0
		if not onlysvd:
			print('No pre loaded data found') 
			file_loc = f'{self.base}/Train'
			
			i = 0
			for subf in os.listdir(file_loc):
				f = f'{file_loc}/{subf}'
				for image in os.listdir(f):
					i = i+1

			train_data = np.zeros((i,size[0],size[1]))
			train_target = np.zeros(i)
			print(i)
			i = 0
			for subf in tqdm(os.listdir(file_loc)):
				f = f'{file_loc}/{subf}'
				#print(subf)
				results = []
				with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
					# Submit tasks to the executor
					futures = [executor.submit(self.process_data,f, image,gray,size,False) for image in os.listdir(f)]
					results =[future.result() for future in futures] 

				train_data[i:i + len(results),:,:] = np.array(results)
				train_target[i:i+len(results)] = int(subf)
				i = i + len(results)
			#print('Data Loaded')
			data = np.reshape(train_data,(train_data.shape[0],train_data.shape[1]*train_data.shape[2]))
			if not os.path.exists(self.train_folder):
				# If it doesn't exist, create the folder
				os.makedirs(self.train_folder)
			np.save(self.fname,data)
			np.save(self.fname_label,train_target)
		else:
			print('Found Pre Loaded Data')
			data = np.load(self.fname)
			train_target = np.load(self.fname_label)
		
		self.train_dat = data
		print(self.train_dat.shape)
		if self.pca_do:
			print('Starting PCA on train')
			self.pca = PCA(n_components=num_components).fit(self.train_dat)
			print('Completed PCA on train')
			self.train_dat = self.pca.transform(self.train_dat)
			print(self.train_dat.shape)
		if self.lda:
			print('Doing LDA')
			self.lda_object = LinearDiscriminantAnalysis()
			self.train_dat = self.lda_object.fit_transform(self.train_dat, train_target)
			print('Done LDA')
		print(self.train_dat.shape)
		self.train_tar = train_target
		self.classes = np.unique(self.train_tar)

	def visualize(self, num = 2):
		if self.lda and self.lda_object is not None:
			plt.figure(figsize=(12, 4))
			plt.suptitle("LDA Eigenvectors")
			print(self.lda_object.scalings_.shape)
			for i in range(num):
				plt.subplot(int(np.ceil(np.sqrt(num)))-1,int(np.ceil(np.sqrt(num))),i+1)
				plt.imshow(self.lda_object.scalings_[:,i].reshape(self.size), cmap = 'gray')
				#plt.imshow(self.lda_object.coef_[i].reshape(self.size), cmap = 'gray')
				plt.title(f'{i+1} eigenvector')

			plt.tight_layout()
			plt.show()
		else:
			if self.pca_do and self.pca is not None:
				plt.figure(figsize=(12, 4))
				plt.suptitle("PCA Eigenvectors")
				for i in range(num):
					plt.subplot(int(np.ceil(np.sqrt(num)))-1,int(np.ceil(np.sqrt(num))),i+1)
					plt.imshow(self.pca.components_[i].reshape(self.size), cmap = 'gray')
					plt.title(f'{i+1} eigenvector')

				plt.tight_layout()
				plt.show()

				print(np.sum(self.pca.explained_variance_))

	def load_data_train(self, do_pca = True, lda = False,size = [64,64], gray = True, num_processes = 4, num_components = 250,
						 fname = 'traindata.npy', fname_label = 'trainlabel.npy'):
		print('Loading Training Data')
		self.size = size
		self.gray = gray
		self.lda = lda
		self.pca_do = do_pca
		onlysvd = False
		self.num_components = num_components
		self.fname = f'{self.train_folder}/{fname}'
		self.fname_label = f'{self.train_folder}/{fname_label}'
		if os.path.isfile(self.fname) and os.path.isfile(self.fname_label):
			onlysvd = True
		self.load_train_mp2(size, gray, num_processes, num_components, onlysvd)
		print("Loaded train_data")

	def load_data_test(self,num_processes = 4):
		self.load_test_mp(self.size, self.gray, num_processes)
		print("Loaded Test data")
		
	def train(self, method = 'kmeans', k = 100):
		self.method = method
		self.model_file = f'{self.model_folder}/{self.method}_{self.num_components}'
		if self.pca_do:
			self.model_file = f'{self.model_file}_pca'
		
		if self.lda:
			self.model_file = f'{self.model_file}_lda'

		if self.method == 'knn':
			self.k = k
			self.model_file = f'{self.model_file}_{k}'

		self.model_file = f'{self.model_file}.joblib'
		
		if os.path.isfile(self.model_file):
			print('Found Pre trained model')
			self.classifier = joblib.load(self.model_file)
			print('Loaded Trained Model')
			if self.method == 'kmeans':
				self.cluster_assignments = self.classifier.labels_
				#print(type(cluster_assignments))
				#print(cluster_assignments.shape)
				#print(np.unique(cluster_assignments))
				print(np.vstack((self.train_tar,self.cluster_assignments)))
		else:
			print('Started Classification training')
			if method == 'kmeans':
				self.classifier = KMeans(n_clusters=self.classes.shape[0], random_state=42)
				self.classifier.fit(self.train_dat)
				self.cluster_assignments = self.classifier.labels_
				#print(type(cluster_assignments))
				#print(cluster_assignments.shape)
				#print(np.unique(cluster_assignments))
				print(np.vstack((self.train_tar,self.cluster_assignments)))

			elif method == 'svc':
				self.classifier = SVC(kernel='rbf', class_weight ='balanced')
				self.classifier.fit(self.train_dat, self.train_tar)
			elif method == 'knn':
				#k = int(np.sqrt(self.train_dat.shape[0]))  # Set the number of neighbors
				print(k)
				self.classifier = KNeighborsClassifier(n_neighbors=k)
				self.classifier.fit(self.train_dat, self.train_tar)
			
			if not os.path.exists(self.model_folder):
    			# If it doesn't exist, create the folder
				os.makedirs(self.model_folder)
			joblib.dump(self.classifier, self.model_file)
			print('Classifier Trained')
	
	def score_test(self):
		if self.method == 'kmeans':
			self.kmeans_transfer()
			
		print('Predictions shape ', self.pred.shape)
		print('Ground Truth shape ', self.test_tar.shape)
		print("Unique values in y_pred:", np.unique(self.pred))
		print("Unique values in y_true:", np.unique(self.test_tar))
		print("Unique Values in train", np.unique(self.train_tar))
		print("Type prediction: ", self.pred.dtype)
		print("Type test: ", self.test_tar.dtype)
		print("Type train: ", self.train_tar.dtype)
		accuracy = accuracy_score(self.test_tar,self.pred)
		cm = confusion_matrix(self.test_tar,self.pred)
		cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
		cm_display.plot(include_values = False)
		if self.pca:
			plt.title(f'{self.method} with PCA')
		else:
			plt.title(f'{self.method} with LDA')
		plt.show()

		if self.method == 'knn':
			print(self.k, accuracy)
		else:
			print(accuracy)
	def kmeans_transfer(self):
		#self.train_tar, self.cluster_assignments, self.pred
		for i in self.classes:
			
			unique_elements, counts = np.unique(self.train_tar[self.cluster_assignments == i], return_counts=True)
			index_of_max_count = np.argmax(counts)
			most_common_value = unique_elements[index_of_max_count]
			self.pred[self.pred == i] = most_common_value

	def predict(self):
		print('Strating Predictions')
		self.pred_name = f'{self.pred_folder}/{self.method}_{self.num_components}'
		
		if self.pca_do:
			self.pred_name = f'{self.pred_name}_pca'

		if self.lda:
			self.pred_name = f'{self.pred_name}_lda'

		if self.method == 'knn':
			self.pred_name  = f'{self.pred_name}_{self.k}'
		
		self.pred_name = f'{self.pred_name}.npy'

		if os.path.isfile(self.pred_name):
			print('Found Predictions')
			self.pred = np.load(self.pred_name)
			print('Loaded Predictions')
		else:
			if not os.path.exists(self.pred_folder):
				# If it doesn't exist, create the folder
				os.makedirs(self.pred_folder)
			
			print('No predictions found, predicting')
			
			print(self.test_dat.shape)
			self.pred = self.classifier.predict(self.test_dat)
			print(self.pred.shape)
			np.save(self.pred_name,self.pred)

			print('predictions completed')



