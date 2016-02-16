import numpy as np
import seaborn as sns
import pylab as pb

def k_se(X,l,sigma2):
    K = np.zeros((X.shape[0],X.shape[0]))
    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[0]):
            K[i,j] = sigma2*np.exp(-1.0/(2.0*l*l)*(X[i,:]-X[j,:]).dot((X[i,:]-X[j,:]).T))
    return K

def kernel_se(xp,xq,l,sigma2):
    #Tile the inputs to make sure the matrices are of the same dimensions
    xp_rep = np.tile(xp,(len(xq),1)).T
    xq_rep = np.tile(xq,(len(xp),1))
    r = (xp_rep - xq_rep)
    r = np.square(r)
    gamma = (-0.5*(l*l))
    kernel = (sigma2*np.exp(gamma*r))
    return kernel

def part1():
    l = 3
    sigma2 = 0.2
    X = np.arange(-3.14159, 3.14159, 0.01)
    X = np.vstack((X,X))
    X = X.T

    covariance = k_se(X,l,sigma2)
    mean = np.zeros(len(X))

    sample_function1 = np.random.multivariate_normal(mean,covariance,1)

    covariance = k_se(X,l,sigma2)
    sample_function2 = np.random.multivariate_normal(mean,covariance,1)

    sns.plt.plot(X,sample_function1.T)
    sns.plt.plot(X,sample_function2.T)
    pb.ylabel("Y")
    pb.xlabel("X")
    pb.xlim(-3.14,3.14)
    pb.title("Functions From Prior")
    pb.savefig(str("prac2_priorfuncs_3l" +'.png'),transparent=True, bbox_inches='tight', pad_inches=0)
    sns.plt.show()


def sinusoidal_model():
    x = np.arange(-1*(np.pi), np.pi, 0.01)
    return np.sin(x)

def draw_lines(x,cov,mean,num_lines=7):
    for line in range(num_lines):
            sample_function = np.random.multivariate_normal(mean,cov,1)
            sns.plt.plot(x,sample_function.T)
    sns.plt.plot(x,mean,linewidth=5.0)
    pb.xlim(-4*3.14,4*3.14)
    pb.savefig(str("prac2_predictpost_1l" +'.png'),transparent=True, bbox_inches='tight', pad_inches=0)
    sns.plt.show()


def part2():
    xp = np.linspace(-np.pi, np.pi, 7)
    xq = np.arange(-4*np.pi, 4*np.pi, 0.1)
    l = 1
    sigma2 = 0.1
    sigma2_noise = 0.5
    y = np.sin(xp)
    e = np.random.normal(0,0.5,(y.shape))
    y = y + e
    k_known = kernel_se(xp,xp,l,sigma2)
    k_mix = kernel_se(xp,xq,l,sigma2)
    k_predict = kernel_se(xq,xq,l,sigma2)
    cov = k_predict - np.dot(np.dot(k_mix.T,np.linalg.inv(k_known + np.eye(k_known.shape[0])*sigma2_noise)),k_mix)
    mean = np.dot(np.dot(k_mix.T,np.linalg.inv(k_known + np.eye(k_known.shape[0])*sigma2_noise)),y)
    draw_lines(xq,cov=cov,mean=mean)
    pb.imshow(cov,interpolation='nearest',cmap='hot')
    pb.savefig('prac2_covariance_1l.png',transparent=True, bbox_inches='tight', pad_inches=0)
    pb.show()




part2()