# Subspace Method for Image Classication

## Required Environment Libraries

- Numpy
- Matplotlib
- scipy
- sklearn
- tqdm
- cv2 
- joblib


## File Structure

- main.py: This is the main file. 

- utils.py: Contains the utility class, that has classes for Image Classification, and Numerical Estimation 


## To run

- To run Image Classification run:
	- For LDA Subspace Method with SVC Classifier:
		- python main.py --qnum 1 --size 32 32 --do_lda --method svc
	- For LDA Subspace Method with knn Classifier:
		- python main.py --qnum 1 --size 32 32 --do_pca --method knn
	- For PCA Subspace Method with SVC Classifier:
		- python main.py --qnum 1 --size 32 32 --do_lda --method svc
	- For PCA Subspace Method with knn Classifier:
		- python main.py --qnum 1 --size 32 32 --do_pca --method knn
- To run numerical estimation run:
	- python main.py --qnum 2 --y_init 0.3 0.2