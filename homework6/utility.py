import numpy as np
from numpy import sqrt, sum, abs, max, maximum, logspace, exp, log, log10, zeros
from numpy.random import normal, randn, choice
from numpy.linalg import norm
from scipy.signal import convolve2d
from scipy.linalg import orth

# For unit testing

def check_gradient(f, grad, x, error_tol = 1e-6):
    y = normal(size=x.shape)
    y = y/norm(y)*norm(x)

    g = grad(x)
    rel_error = np.zeros(10)
    for iter in range(10):
        y = y/10;
        d1 = f(x + y) - f(x)  # exact change in function
        d2 = np.sum((g * y).ravel())  # approximate change from gradient
        rel_error[iter] = (d1-d2)/d1
        #print('d1=%1.5g, d2=%1.5g, error=%1.5g,'%(d1,d2, rel_error[iter] ))
    min_error = min(np.abs(rel_error))
    print('Min relative error = %1.5g'%min_error)
    did_pass = min_error < error_tol
    print(did_pass and "Test passed" or "Test failed")
    return did_pass

def check_adjoint(A,At,dims):
    # start with this line - create a random input for A()
    x = normal(size=dims)+1j*normal(size=dims)
    Ax = A(x)
    y = normal(size=Ax.shape)+1j*normal(size=Ax.shape)
    Aty = At(y)
    # compute the Hermitian inner products
    inner1 = np.sum(np.conj(Ax)*y)
    inner2 = np.sum(np.conj(x)*Aty)
    # report error
    rel_error = np.abs(inner1-inner2)/np.maximum(np.abs(inner1),np.abs(inner2))
    if rel_error < 1e-10:
        print('Adjoint Test Passed, rel_diff = %s'%rel_error)
        return True
    else:
        print('Adjoint Test Failed, rel_diff = %s'%rel_error)
        return False

# For total-variation

kernel_h = [[1,-1,0]]
kernel_v = [[1],[-1],[0]]

# Do not modify ANYTHING in this cell.
def gradh(x):
    """Discrete gradient/difference in horizontal direction"""
    return convolve2d(x,kernel_h, mode='same', boundary='wrap')
def gradv(x):
    """Discrete gradient/difference in vertical direction"""
    return convolve2d(x,kernel_v, mode='same', boundary='wrap')
def grad2d(x):
    """The full gradient operator: compute both x and y differences and return them all.  The x and y
    differences are stacked so that rval[0] is a 2D array of x differences, and rval[1] is the y differences."""
    return np.stack([gradh(x),gradv(x)])

def gradht(x):
    """Adjoint of gradh"""
    kernel_ht = [[0,-1,1]]
    return convolve2d(x,kernel_ht, mode='same', boundary='wrap')
def gradvt(x):
    """Adjoint of gradv"""
    kernel_vt = [[0],[-1],[1]]
    return convolve2d(x,kernel_vt, mode='same', boundary='wrap')
def divergence2d(x):
    "The methods is the adjoint of grad2d."
    return gradht(x[0])+gradvt(x[1])


# For logistic regression

def buildmat(m, n, cond_number):
    """Build an mxn matrix with condition number cond."""
    if m <= n:
        U = randn(m, m);
        U = orth(U);
        Vt = randn(n, m);
        Vt = orth(Vt).T;
        S = 1 / logspace(0, log10(cond_number), num=m);
        return (U * S[:, None]).dot(Vt)
    else:
        return buildmat(n, m, cond_number).T


def create_classification_problem(num_data, num_features, cond_number):
    """Build a simple classification problem."""
    X = buildmat(num_data, num_features, cond_number)
    # The linear dividing line between the classes
    w = randn(num_features, 1)
    # create labels
    prods = X @ w
    y = np.sign(prods)
    #  mess up the labels on 10% of data
    flip = choice(range(num_data), int(num_data / 10))
    y[flip] = -y[flip]
    #  return result
    return X, y


def logistic_loss(z):
    """Return sum(log(1+exp(-z))). Your implementation can NEVER exponentiate a positive number.  No for loops."""
    loss = zeros(z.shape)
    loss[z >= 0] = log(1 + exp(-z[z >= 0]))
    # Make sure we only evaluate exponential on negative numbers
    loss[z < 0] = -z[z < 0] + log(1 + exp(z[z < 0]))
    return np.sum(loss)


def logreg_objective(w, X, y):
    """Evaluate the logistic regression loss function on the data and labels, where the rows of D contain
    feature vectors, and y is a 1D vector of +1/-1 labels."""
    z = y * (X @ w)
    return logistic_loss(z)


def logistic_loss_grad(z):
    """Gradient of logistic loss"""
    grad = zeros(z.shape)
    neg = z.ravel() <= 0
    pos = z.ravel() > 0
    grad[neg] = -1 / (1 + exp(z[neg]))
    grad[pos] = -exp(-z[pos]) / (1 + exp(-z[pos]))
    return grad


def logreg_objective_grad(w, X, y):
    return X.T @ (y * logistic_loss_grad(y * X @ w))

# For gradient descent
def estimate_lipschitz(g, x):
    # Your work here
    y = x+normal(size=x.shape)
    L = norm(g(x)-g(y))/norm(x-y)
    return L
