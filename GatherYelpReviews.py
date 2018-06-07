## tools for computational exercises in MS&E 111/211
## author: Benjamin Van Roy

import numpy as np
from scipy import optimize
import pandas as pd
from cvxopt import matrix,solvers
from tabulate import tabulate

def lineqsolve(A, b, print_message=True):
	'''
	arguments
	A: mxn np.matrix
	b: mx1 np.matrix
	print_message: whether to print a message pertaining to success/failure of the code

	effects
	solves linear system of equations: Ax = b
	if there is no solution, return a vector that minimizes the Euclidean norm of Ax - b
	if there are multiple solutions, provide the one with minimal Euclidean norm

	returns
	x: optimal solution as an nx1 np.matrix
	solved: boolean indicating whether Ax = b
	'''
	x,residual,_,_ = np.linalg.lstsq(A,b)
	solved = not bool(residual.sum())
	if print_message:
		if solved:
			print "solved"
		else:
			print "no solution"
	return x,solved


def linprog(A, b, c, A_eq=None, b_eq=None, lb=None, ub=None, form='symmetric', print_message=True, solver='scipy'):
	'''
	arguments
	A: mxn np.matrix
	b: mx1 np.matrix
	c: 1xn np.matrix
	A_eq: kxn np.matrix
	b_eq: kx1 np.matrix
	lb: nx1 np.matrix
	ub: nx1 np.matrix
	form: "symmetric" for <= constraints, "standard" for = constraints
	print_message: whether to print a message pertaining to success/failure of the code
	solver: which solver to use either 'scipy' or 'cvxopt' only changes affect algorithm used
		if using form='general'

	effects
	solves optimization problem taking one of the following two forms:
	symmetric: max c x s.t. Ax <= b, x >= 0
	standard: max c x s.t. Ax = b, x >= 0
    general: max cx s.t. A x <= b, A_eq x = b_eq, lb <= x <= ub

	returns
	x: optimal solution as an nx1 np.matrix
	objective_value: optimal objective value
	'''
	print c.shape
	if form=='symmetric':
		sol = optimize.linprog(c=-np.array(c)[0], A_ub=A, b_ub=np.array(b.T)[0], options={'bland': True})
	elif form=='standard':
		sol = optimize.linprog(c=-np.array(c)[0], A_eq=A, b_eq=np.array(b.T)[0], options={'bland': True})
	elif form=='general':
		if (lb is None) and (ub is not None):
			lb = -np.inf*np.ones(ub.shape)
		if (ub is None) and (lb is not None):
			ub = np.inf*np.ones(lb.shape)
		if (ub is not None) and (lb is not None):
			bounds = zip([lb[i,0] for i in range(len(lb))], [ub[i,0] for i in range(len(ub))])
		else:
			bounds = (None, None)
		if solver == 'cvxopt':
			n = A.shape[1]
			if not print_message:
				solvers.options['show_progress'] = False
			Atemp = A if not A is None else np.zeros((1, n))
			A_eqtemp = A_eq if not A_eq is None else np.zeros((1, n))
			combo = np.concatenate((Atemp, A_eqtemp), axis=0)
			idx = np.where(np.sum(combo != 0, axis=0) > 0)[1]
			x = np.zeros((n,1))
			A = matrix(A[:,idx].astype(float)) if not A is None else None
			b = matrix(b.astype(float)) if not b is None else None
			A_eq = matrix(A_eq[:,idx].astype(float)) if not A_eq is None else None
			b_eq = matrix(b_eq.astype(float)) if not b_eq is None else None
			sol = solvers.lp(matrix(-c[:,idx].T.astype(float)), A, b, A_eq, b_eq)
			x[idx] = np.array(sol['x'])
			return x, -sol['primal objective']
		sol = optimize.linprog(c=-np.array(c)[0],
							   A_ub=A,
							   b_ub=np.array(b.T)[0] if not b is None else None,
							   A_eq=A_eq,
							   b_eq=np.array(b_eq.T)[0] if not b_eq is None else None,
							   bounds=bounds,
							   options={'bland': True})
	else:
		print 'ERROR: unrecognized LP form ' + form
		return np.nan, np.nan
	if print_message:
		print sol['message']
	x = np.matrix(sol['x']).T
	objective_value = -sol['fun']
	return x, objective_value


def quadprog(A, b, c, H, print_message=True):
	'''
	arguments
	A: mxn np.matrix
	b: mx1 np.matrix
	c: 1xn np.matrix
	H: nxn matrix
	print_message: whether to print a message pertaining to success/failure of the code

	effects
	solves optimization problem of the form: max c x - x.T H x s.t. Ax <= b

	returns
	x: optimal solution as an nx1 np.matrix
	objective_value: optimal objective value
	'''
	solvers.options['show_progress'] = True
	#solvers.options['feastol']=2e-2
	sol = solvers.qp(P=matrix(2.0*H), q=matrix(-1.0*c.T), G=matrix(1.0*A), h=matrix(1.0*b))
	if print_message:
		print sol['status']
	x = np.matrix(sol['x'])
	objective_value = -sol['primal objective']
	return x, objective_value


def quadprog2(A, b, c, H, print_message=True):
	'''
	arguments
	A: mxn np.matrix
	b: mx1 np.matrix
	c: 1xn np.matrix
	H: nxn matrix
	print_message: whether to print a message pertaining to success/failure of the code

	effects
	solves optimization problem of the form: max c x - x.T H x s.t. Ax <= b

	returns
	x: optimal solution as an nx1 np.matrix
	objective_value: optimal objective value
	'''
	# find initial solution
	x0,obj0 = quadprog(A, b, 0.0*c, 0.0*H, print_message=False)

	f = lambda x: float(-np.matrix(c)*np.matrix(x).T + np.matrix(x) * np.matrix(H) * np.matrix(x).T)
	jac = lambda x: -np.array(c)[0] + 2.0 * np.array((np.matrix(H) * np.matrix(x).T).T)[0]
	hess = lambda x: 2.0 * np.matrix(H)
	g = lambda x: np.array((np.matrix(b).T - np.matrix(A) * np.matrix(x).T).T)[0]
	constraints = {'type':'ineq', 'fun':g, 'jac': lambda x: -A}
	options = {'maxiter': 1e6, 'disp': print_message}
	sol = optimize.minimize(f, x0, method='SLSQP', jac=jac, hess=hess, constraints=constraints, options=options)
	x = np.matrix(sol['x']).T
	objective_value = -sol['fun']
	return x, objective_value


def load_matrix(filepath):
	'''
	arguments
	filepath: full path/filename of the csv source data file

	effects
	reads data from a csv file

	returns
	M: an mxn np.matrix with the numerical data from the file
	row_labels: a list of length m
	column_labels: a list of length n
	'''
	df = pd.read_csv(filepath, index_col=0)
	row_labels = df.index.tolist()
	column_labels = df.columns.tolist()
	M = np.matrix(df)
	return M, row_labels, column_labels


def save_matrix(filepath, M, row_labels=None, column_labels=None):
	'''
	arguments
	filepath: full path/filename of the target csv file
	M: mxn np.matrix
	row_labels: None or a list of length m
	column_labels: None or a list of length n

	effects
	saves the data as a csv file
	'''
	nrows,ncols = M.shape
	if column_labels is None:
		column_labels = range(1,ncols+1)
	if row_labels is None:
		row_labels = range(1,nrows+1)

	df = pd.DataFrame(data=M, index=row_labels, columns=column_labels)
	df.to_csv(filepath, index_label='labels')


def print_matrix(M,row_labels=None,column_labels=None):
	'''
	arguments
	M: mxn np.matrix
	row_labels: None or a list of length m
	column_labels: None or a list of length n

	effects
	prints a table populated with elements of the matrix, with the provided labels
	'''
	nrows,ncols = M.shape
	headers = ['labels'] + (column_labels if column_labels is not None else range(1,ncols+1))
	if row_labels is None:
		row_labels = range(1,nrows+1)
	M_list = M.tolist()
	table = [[row_labels[idx]] + M_list[idx] for idx in range(nrows)]
	print ''
	print tabulate(table, headers=headers)


def basic_simplex(A, b, c, basis):
	'''
	arguments
	A: mxn np.matrix
	b: mx1 np.matrix
	c: 1xn np.matrix
	basis: 1xn np.matrix with boolean values,
		basic variables specifying an initial basic feasible solution

	effects
	solves standard form linear program: max c x s.t. Ax = b, x >= 0

	returns
	x: optimal solution as an nx1 np.matrix
	objective_value: optimal objective value
	'''
	r = np.matlib.ones(c.shape)
	while True:
		# create lists of basic and nonbasic variable indices
		basic = [idx for idx in range(len(basis.T)) if basis[0,idx]]
		nonbasic = [idx for idx in range(len(basis.T)) if ~basis[0,idx]]

		# compute reduced profits
		r[:,nonbasic] = c[:,nonbasic] - c[:,basic] * lineqsolve(A[:,basic], A[:,nonbasic], False)[0]
		r[:,basic] = 0;

		# select nonbasic variable with large reduced profits
		r_max = np.max(r)
		r_argmax = np.argmax(r)
		if r_max > 0:
			numerator = lineqsolve(A[:,basic], b)[0]
			denominator = lineqsolve(A[:,basic], A[:,r_argmax], False)[0]
			alpha = np.matlib.zeros(numerator.shape)
			# resolve division by zero
			for l in range(len(numerator)):
				if denominator[l] == 0:
					alpha[l] = np.inf
				else:
					alpha[l] = numerator[l]/denominator[l]

			# Bland's rule: alpha should be positive
			#     take sign(0/a) to be sign(a)
			#     so if a<0 then a is ruled out
			sign = (lineqsolve(A[:,basic], A[:,r_argmax], False)[0] > 0)
			minval = np.inf

			# select basic variable that is first to become zero
			for l in range(len(alpha)):
				# The condition "sign(l) > 0 is added to rule out the case of 0/negative
				if alpha[l] >= 0 and alpha[l] < minval and sign[l] > 0:
					lmin = l
					minval = alpha[l]

			# swap basic <--> nonbasic variables
			basis[:,basic[lmin]] = False
			basis[:,r_argmax] = True
		else:
			break

	x = np.zeros(c.shape).T
	x[basic,:] = lineqsolve(A[:,basic], b)[0]
	x[nonbasic,:] = 0

	objective_value = float(c*x)
	return x, objective_value



def min_cost_flow(c, u, b):
	'''
	arguments
	c: nxn np.matrix
	u: nxn np.matrix
	b: nx1 np.matrix


	effects
	For a flow network with n nodes. It solves the minimum cost network flow
	problem as described in the lecture. The c matrix should be organized such
	that c[i, j] corresponds to the cost of sending 1 unit of flow from i to j.
	Likewise, u is defined. The b vector is defined as a the net flow produced
	by node j is b_j.

	returns
	f: nxn np.matrix the flows of the network
	cost: minimum cost
	'''
	n = c.shape[0]
	c = c.reshape((1,-1))
	ub = u.reshape((-1,1))
	lb = np.zeros((n**2, 1))
	b_eq = b
	A1 = np.repeat(np.identity(n), n, axis=1)
	A2 = np.tile(np.identity(n), (1,n))
	A_eq = A1 - A2
	x, objective_value = linprog(np.zeros((1,n**2)), np.zeros((1,1)), -c, A_eq, b_eq, lb, ub, 'general')
	f = x.reshape((n,n))
	return f, -objective_value


def logistic_regression_newtons_method(X, Y_onehot, num_iters):
        '''
        arguments
        X: JxN np.array
        Y_onehot: JxC np.matrix (in one-hot form)
        num_iters: scalar

        effects
        runs logistic regression using newton's method for num_iter iterations

        returns
        x: optimal solution (i.e. resulting parameters)
        errors: misclassification errors at each iteration
        '''
        J, N = X.shape
        _, C = Y_onehot.shape
        beta = np.zeros(((C-1)*(N+1), 1)) # initialize the parameters
        X = np.concatenate((-np.ones((J, 1)), X), axis=1) # add the intercept
        errors = np.zeros((num_iters,1))

        ######## create X_tilde
        X_tilde = np.zeros((J*(C-1), (N+1)*(C-1)))
        for i in range(C-1):
                X_tilde[i*J:(i+1)*J, i*(N+1):(i+1)*(N+1)] = X
        ######## create y
        y = np.zeros((J*(C-1),1))
        for l in range(C-1):
                for i in range(J):
                        if Y_onehot[i,l] == 1:
                                y[l*J+i] = 1

        iter_no = 0
        while iter_no < num_iters:
                iter_no += 1

                ######## update P
                beta_matrix = np.reshape(beta, ((C-1), (N+1)))
                P_matrix = compute_probs(X[:,1:], beta_matrix, C).T
                P = np.reshape(P_matrix, (J*(C-1), 1))

                ######## update W
                W = np.zeros((J*(C-1), J*(C-1)))
                for k in range(C-1):
                        for m in range(C-1):
                                W_km = np.zeros((J,J))
                                for i in range(J):
                                        if k == m:
                                                W_km[i,i] = P_matrix[k,i] * (1-P_matrix[k,i])
                                        else:
                                                W_km[i,i] = -P_matrix[k,i] * P_matrix[m,i]
                                W[k*J:(k+1)*J, m*J:(m+1)*J] = W_km

                ######## update beta
                gradient = np.matmul(X_tilde.T, (y-P))
                hessian = -np.matmul(np.matmul(X_tilde.T, W), X_tilde)
                beta -= np.matmul(np.linalg.inv(hessian), gradient)

                ######## compute misclassification error
                y_pred = predict(X[:,1:], Y_onehot, beta)
                Y = np.reshape(np.argmax(Y_onehot, axis=1), (J,1))
                errors[iter_no-1, 0] = 1.0*np.sum(Y != y_pred) / J

        x = beta.copy()
        return x, errors



def logistic_regression_gradient_descent(X, Y_onehot, num_iters, learning_rate=0.01):
        '''
        arguments
        X: JxN np.array
        Y_onehot: JxC np.matrix (in one-hot form)
        num_iters: scalar
        learning_rate: scalar

        effects
        runs logistic regression using gradient descent method for num_iter iterations

        returns
        x: optimal solution (i.e. resulting parameters)
        errors: misclassification errors at each iteration
        '''
        J, N = X.shape
        _, C = Y_onehot.shape
        beta = np.zeros(((C-1)*(N+1), 1)) # initialize the parameters
        X = np.concatenate((-np.ones((J, 1)), X), axis=1) # add the intercept
        errors = np.zeros((num_iters,1))

        ######## create X_tilde
        X_tilde = np.zeros((J*(C-1), (N+1)*(C-1)))
        for i in range(C-1):
                X_tilde[i*J:(i+1)*J, i*(N+1):(i+1)*(N+1)] = X
        ######## create y
        y = np.zeros((J*(C-1),1))
        for l in range(C-1):
                for i in range(J):
                        if Y_onehot[i,l] == 1:
                                y[l*J+i] = 1

        iter_no = 0
        while iter_no < num_iters:
                iter_no += 1

                ######## update P
                beta_matrix = np.reshape(beta, ((C-1), (N+1)))
                P_matrix = compute_probs(X[:,1:], beta_matrix, C).T
                P = np.reshape(P_matrix, (J*(C-1), 1))

                ######## update beta
                gradient = np.matmul(X_tilde.T, (y-P))
                beta += learning_rate * gradient

                ######## compute misclassification error
                y_pred = predict(X[:,1:], Y_onehot, beta)
                Y = np.reshape(np.argmax(Y_onehot, axis=1), (J,1))
                errors[iter_no-1, 0] = 1.0*np.sum(Y != y_pred) / J

        x = beta.copy()
        return x, errors




def compute_probs(X, beta_matrix, C):
        '''
        helper function

        returns
        Jx(C-1) dimensional probability matrix
        entry in the r'th row, c'th column is the probability that observation r
        belongs to class c
        
        effects
        computes the probability values given the parameters beta
        uses log trick for numerical stability
        '''
        J, N = X.shape
        X_tmp = np.concatenate((-np.ones((J, 1)), X), axis=1) # add the intercept
        P_matrix = np.zeros(((C-1), J))
        for i in range(J): # iterate over columns first
                beta_matrix_conc = np.concatenate((np.zeros((1,N+1)),beta_matrix), axis=0)
                exponent = np.sum(np.multiply(beta_matrix_conc, X_tmp[i,:]), axis=1)
                a = np.amax(exponent)
                denominator_log = a + np.log(np.sum(np.exp(exponent-a), axis=0))
                numerator_log = np.sum(np.multiply(beta_matrix, X_tmp[i,:]), axis=1)
                tmp_log = numerator_log - denominator_log
                P_matrix[:,i] = np.exp(tmp_log)

        return P_matrix.T



def predict(X, Y_onehot, x):
        '''
        helper function
        x: (C-1)*(N+1) x 1 dimensional
        you can directly use the output parameters x of logistic regression function
        
        effects
        returns the predicted labels given the parameters x
        '''
        J,C = Y_onehot.shape
        _,N = X.shape
        beta_matrix = np.reshape(x, ((C-1), (N+1)))
        probs = compute_probs(X, beta_matrix, C)
        last_clmn = np.ones((J,1)) - np.sum(probs, axis=1, keepdims=True) # the prob values of the last class
        probs = np.concatenate((probs, last_clmn), axis=1)

        y_pred = np.argmax(probs, axis=1)
        y_pred = np.reshape(y_pred, (J, 1))
        return y_pred


def normalize_data(X, X_test=None):
        '''
        helper function
            
        effects
        normalizes the training data X and test data X_test (if given)
        '''
        ############# normalize training data X
        mean = np.mean(X, axis=0)
        X -= mean
        stddev = np.std(X, axis=0)
        X /= stddev
        ############# normalize test data if given
        if X_test is not None:
                X_test -= mean
                X_test /= stddev
                
        return X, X_test


def convert_to_onehot(Y, C):
        '''
        helper function
        (classes must be represented using integers 0,1,2,...)
        
        effects
        converts Y to one-hot vector representation
        '''
        Y_onehot = np.zeros((len(Y), C))
        for i in range(Y_onehot.shape[1]):
                Y_onehot[Y[:,0]==i, i] = 1

        return Y_onehot
