# This is the main file for HW 4

import numpy as np
import argparse
from utils import integrate, Classifier


def ques1():
	classify = Classifier(opt.root)
	classify.load_data_train(do_pca = opt.do_pca, lda = opt.do_lda, size = opt.size, num_components = opt.num_components)
	classify.load_data_test()
	classify.train(method = opt.method, k = opt.k_neighbours)
	classify.predict()
	classify.score_test()
	if opt.visualize:
		classify.visualize()	

def ques2():
	step_size = opt.step_size
	start_time = opt.start_time
	end_time = opt.end_time
	y_init = opt.y_init

	steps = np.ceil((end_time - start_time)/step_size) + 1
	x_array = np.linspace(start_time, end_time,int(steps))
	y_init = np.array(y_init).reshape((len(y_init),1))

	integr = integrate(y_init,x_array,step_size)
	integr.integrate()
	integr.plot()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qnum', type=int, default=2, help='Question Number')
    parser.add_argument('--step_size', type = float, default = 0.05)
    parser.add_argument('--start_time', type = int, default = 0)
    parser.add_argument('--end_time', type = int, default = 30)
    parser.add_argument('--y_init', nargs='+', type = float)
    parser.add_argument('--size', nargs='+', type = int)
    parser.add_argument('--num_components', default = 64)
    parser.add_argument('--method', default = 'knn')
    parser.add_argument('--root', default = './Data')
    parser.add_argument('--k_neighbours', default = 6)
    parser.add_argument('--do_pca', action = 'store_true')
    parser.add_argument('--do_lda', action = 'store_true')
    parser.add_argument('--visualize', action = 'store_true')



    opt = parser.parse_args()
    if opt.qnum == 2:
    	ques2()
    elif opt.qnum == 1:
    	ques1()