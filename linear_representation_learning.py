import numpy as np
import scipy.optimize as opt
import seaborn as sns
import pylab as pb
noise_sigma2 = 0

# Global state
def f(W):
    # d: dimension of weights
    # n: number of weights
    # noise_sigma2: variance to be used for observation noise
    # return the value of the objective at x
    W = W.reshape((10,2))
    n = 100
    d = 10
    # noise_sigma2 = 0
    C = np.dot(W,W.T) + noise_sigma2*np.eye(W.shape[0])
    # C = C/n
    C_inv = np.linalg.inv(C)
    S = np.dot(y,y.T)
    S = S
    A = 0.5*n*np.log(np.linalg.det(C))
    B = 0.5*np.trace(np.dot(C_inv,S))
    const = n*d*np.log(2*noise_sigma2)
    W = W.flatten()
    val = A + B + const
    return val

def f_nonlin(x):
    sinus = np.sin(x)
    cosinus = np.cos(x)
    sinus = np.multiply(x,sinus)
    cosinus = np.multiply(x,cosinus)
    return np.vstack((sinus,cosinus))

def f_lin(X,W):
    return np.dot(W,X)

def J_mat(W,i,j):
    J = np.zeros((W.shape[0],W.shape[1]))
    J[i][j] = 1
    return J

def grad_ww(W,i,j):
    J = J_mat(W,i,j)
    return np.dot(J,W.T) + np.dot(W,J.T)

def dfx(W):
    W = W.reshape((10,2))
    n = 100
    d = 10
    C = np.dot(W,W.T) + noise_sigma2*np.eye(W.shape[0])
    # C = C/n
    C_inv = np.linalg.inv(C)
    derivatives=np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            dB = 0.5*(np.trace(np.dot(np.dot(y,y.T),(-1)*np.dot(np.dot(C_inv,grad_ww(W,i,j)),C_inv))))
            dA = (n/2.0)*np.trace(np.dot(C_inv,grad_ww(W,1,1)))
            grad = dB + dA
            derivatives[i][j] = grad
    W = derivatives
    W = W.flatten()
    return W

W = np.random.normal(0,1,(10,2))
x = np.linspace(0,4*np.pi,100)
y_nonlin = f_nonlin(x)
y = f_lin(y_nonlin,W)

#### F()
W1 = np.random.normal(0,1,(10,2))
# print(f(W1))
# dfx(A,B,const,y,W,C_inv)
# print(W)
W = W.flatten()
guess = np.random.normal(0,1,(10,2))
guess = guess.flatten()
x_star = opt.fmin_cg(f,guess,fprime=dfx)
print(x_star.shape)
x_star = x_star.reshape((10,2))
x_guess = np.linalg.lstsq(x_star, y)
x1 = np.asarray(x_guess[0][0])
x2 = np.asarray(x_guess[0][1])
# print(x_star)
# print(x_guess[0][0])
# x_guess = np.asarray(x_guess)
# print(x_guess.shape)
sns.plt.plot(x1,x2)
pb.title("Recovered Generating Parameters")
pb.ylabel("X1")
pb.ylabel("X2")
pb.savefig('prac3_recovered.png',transparent=True, bbox_inches='tight', pad_inches=0)
sns.plt.show()
sns.plt.plot(y_nonlin[0,:],y_nonlin[1,:])
pb.title("True Generating Parameters")
pb.ylabel("X1")
pb.ylabel("X2")
pb.savefig('prac3_true.png',transparent=True, bbox_inches='tight', pad_inches=0)
sns.plt.show()