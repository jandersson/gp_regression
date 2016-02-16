import pylab as pb
from numpy import array, arange, dot, ones, random, eye, linalg
import seaborn as sns

def add_noise(X, mean=0, stddev=0.54):
    e = random.normal(mean, stddev, len(X))
    X = X + e
    return X

def linear_model(X,a):
    return dot(X.T,a)

def compute_prior(alpha=2, num_samples=100000):
    I = eye(len(a))
    prior_cov = (alpha**-1)*I
    prior_mean = array([0,0])
    w_prior = random.multivariate_normal(prior_mean,prior_cov,num_samples)
    # pb.savefig('img/prac1_prior.png',transparent=True, bbox_inches='tight', pad_inches=0)
    plot2DGaussian(w_prior,1,title="Prior",filename='prac1_prior')
    return w_prior

def plot2DGaussian(X,fid,title="Plot",filename="random_plot"):
    # pb.figure(fid)
    sns.jointplot(X[:,0],X[:,1],kind='hex',xlim=(-3,3),ylim=(-3,3))
    pb.title(title)
    pb.savefig(str(filename +'.png'),transparent=True, bbox_inches='tight', pad_inches=0)
    sns.plt.show()

def compute_posterior(X,Y,variance=0.3, num_samples=100000):
    #X is an input vector D x n where n is the number of observations and D is the dimension
    #Y is a vector of scalar outputs or target (dependent variable)
    alpha = 2.0
    beta = 25
    A = beta*dot(X,X.T) + alpha*eye(len(X))
    A_inv = linalg.inv(A)
    mean = beta*(dot(dot(A_inv,X),Y))
    w_posterior = random.multivariate_normal(mean,A_inv,num_samples)
    return w_posterior

def plot_functions(samples,title="Random Plot",filename="random_plot"):
    for sample in w_samples:
        y = linear_model(X_offset,sample)
        sns.plt.plot(X,y)

    pb.ylabel("Y")
    pb.xlabel("X")
    pb.title(title)
    pb.savefig(str(filename +'.png'),transparent=True, bbox_inches='tight', pad_inches=0)



a = array([-1.3, 0.5])
X = arange(-1,1.01,0.01)
X_offset = array([ones(len(X)), X])
data = linear_model(X_offset,a)
target = add_noise(data)

num_samples = 100000
w_prior = compute_prior()

#Plotting functions from the prior
w_samples = w_prior[:6,:]

# for sample in w_samples:
#     y = linear_model(X_offset,sample)
#     sns.plt.plot(X,y)

datapoint_x = X_offset[:,:1]
datapoint_y = target[:1]
w_posterior = compute_posterior(datapoint_x,datapoint_y)
plot2DGaussian(w_posterior,2,title="Posterior",filename="prac1_posterior_1pt")
w_samples = w_posterior[:15,:]
plot_functions(w_samples,title="Posterior Samples",filename="prac1_postfuncs_1pt")

datapoint_x = X_offset[:,:6]
datapoint_y = target[:6]
w_posterior = compute_posterior(datapoint_x,datapoint_y)
plot2DGaussian(w_posterior,2,title="Posterior",filename="prac1_posterior_6pt")
w_samples = w_posterior[:15,:]
plot_functions(w_samples,title="Posterior Samples",filename="prac1_postfuncs_6pt")
# w_samples = w_posterior[:6,:]
# for sample in w_samples:
#     y = linear_model(X_offset,sample)
#     sns.plt.plot(X,y)
# sns.plt.figure(1)
# sns.plt.plot(X,target)

datapoint_x = X_offset[:,:100]
datapoint_y = target[:100]
w_posterior = compute_posterior(datapoint_x,datapoint_y)
plot2DGaussian(w_posterior,2,title="Posterior",filename="prac1_posterior_100pt")
w_samples = w_posterior[:15,:]
plot_functions(w_samples, title="Posterior Samples",filename="prac1_postfuncs_100pt")
# w_samples = w_posterior[:6,:]
# for sample in w_samples:
#     y = linear_model(X_offset,sample)
#     sns.plt.plot(X,y)
# sns.plt.figure(1)
# sns.plt.plot(X,target)

datapoint_x = X_offset[:,:]
datapoint_y = target[:]
w_posterior = compute_posterior(datapoint_x,datapoint_y)
w_samples = w_posterior[:6,:]
plot2DGaussian(w_posterior,2,title="Posterior",filename="prac1_posterior_Allpt")
w_samples = w_posterior[:15,:]
plot_functions(w_samples,title="Posterior Samples",filename="prac1_postfuncs_Allpt")
# for sample in w_samples:
#     y = linear_model(X_offset,sample)
#     sns.plt.plot(X,y)
# sns.plt.figure(1)
# my_plot = sns.plt.plot(X,target)
# my_plot.axes.set_ylim(-1,1)
# my_plot.axes.set_xlim(-1,1)


