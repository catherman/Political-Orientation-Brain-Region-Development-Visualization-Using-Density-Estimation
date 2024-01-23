#!/usr/bin/env python
# coding: utf-8

# 2. Density estimation: Psychological experiments. [45 points]
# In Kanai, R., Feilden, T., Firth, C. and Rees, G., 2011. Political orientations are correlated
# with brain structure in young adults. Current biology, 21(8), pp.677-680., data are collected
# to study whether or not the two brain regions are likely to be independent of each other
# and considering di erent types of political view For this question; you can use third
# party histogram and KDE packages; no need to write your own. The data set
# n90pol.csv contains information on 90 university students who participated in a psychological
# experiment designed to look for relationships between the size of di erent regions of the brain
# and political views. The variables amygdala and acc indicate the volume of two particular
# brain regions known to be involved in emotions and decision-making, the amygdala and the
# anterior cingulate cortex; more exactly, these are residuals from the predicted volume, after
# adjusting for height, sex, and similar body-type variables. The variable orientation gives the
# students' locations on a  five-point scale from 1 (very conservative) to 5 (very liberal). Note
# that in the dataset, we only have observations for orientation from 2 to 5.
# 
# Recall in this case, the kernel density estimator (KDE) for a density is given by
# 
# $$p(x) = \frac{1}{m}\sum_{i=1}^{m}\frac{1}{h}K(\frac{x^{i}-x}{h})$$.
# 
# where $x^{i}$ are two-dimensional vectors, h > 0 is the kernel bandwidth, based on the criterion
# we discussed in lecture. For one-dimensional KDE, use a one-dimensional Gaussian kernel
# 
# $$K(x) = \frac{1}{\sqrt{2\pi}} e^{-x^{2}/2}$$
# 
# For two-dimensional KDE, use a two-dimensional Gaussian kernel: for
# 
# 
# $$x=\begin{bmatrix}x_{1}\\ x_{2}\end{bmatrix}\varepsilon R^{2}$$,
# 
# 
# $$x=\begin{bmatrix}x_{1}\\ x_{2}\end{bmatrix}\epsilon R^{2}$$,
# 
# where x1 and x2 are the two dimensions respectively
# 
# $$K(x) = \frac{1}{2\pi}e^{-\frac{(x_{1})^{2}+(x_{2})^{2}}{2}}$$
# 
#     (a) (5 points) Form the 1-dimensional histogram and KDE to estimate the distributions
#     of amygdala and acc, respectively. For this question, you can ignore the variable 
#     orienta-tion. Decide on a suitable number of bins so you can see the shape of the 
#     distribution clearly. Set an appropriate kernel bandwidth h > 0.
# 
#     (b) (5 points) Form 2-dimensional histogram for the pairs of variables (amygdala,
#     acc). Decide on a suitable number of bins so you can see the shape of the distribution
#     clearly.
# 
#     (c) (20 points) Use kernel-density-estimation (KDE) to estimate the 2-dimensional 
#     density function of (amygdala, acc) (this means for this question, you can ignore the
#     variable orientation). Set an appropriate kernel bandwidth h > 0.
#     Please show the two-dimensional KDE (e.g., two-dimensional heat-map, two-dimensional
#     contour plot, etc.) Please explain what you have observed: is the distribution 
#     unimodal or bi-modal? Are there any outliers? Are the two variables (amygdala, acc)
#     likely to be independent or not? (NOTE:  It actually involves prerequisite knowledge. From traditional probability and statistics, how do you show that two random variables are independent? Once you can answer that, then you need to visually represent that rule somehow and test for independence. )  Please support your argument with reasonable 
#     investigations.
# 
#     (d) (10 points) We will consider the variable orientation and consider conditional 
#     distributions. Please plot the estimated conditional distribution of amygdala 
#     conditioning on political orientation: p(amygdala j orientation = c), c = 2; : : : ; 5, 
#     using KDE. Set an appropriate kernel bandwidth h > 0. Do the same for the volume of
#     the acc: plot
#     p(accjorientation = c), c = 2; : : : ; 5 using KDE. (Note that the conditional 
#     distribution can be understood as  tting a distribution for the data with the same
#     orientation. Thus you should plot 8 one-dimensional distribution functions in total 
#     for this question.)Now please explain based on the results, can you infer that the
#     conditional distribution of amygdala and acc, respectively, are di erent from c = 2; :
#     : : ; 5? This is a type of
#     scienti c question one could infer from the data: Whether or not there is a di erence
#     between brain structure and political view.
#     
#     
#     Now please also  ll out the conditional sample mean for the two variables:
# 
# 
# 
#     Remark: As you can see this exercise, you can extract so much more information from
#     density estimation than simple summary statistics (e.g., the sample mean) in terms of
#     explorable data analysis.
# 
#     (e) (5 points) Again we will consider the variable orientation. We will estimate the
#     conditional joint distribution of the volume of the amygdala and acc, conditioning on
#     a function of political orientation: p(amygdala; accjorientation = c), c = 2; : : : ;
#     5. You will use two-dimensional KDE to achieve the goal; et an appropriate kernel
#     band-width h > 0. Please show the two-dimensional KDE (e.g., two-dimensional heat-map,
#     two-dimensional contour plot, etc.).Please explain based on the results, can you infer
#     that the conditional distribution of
#     two variables (amygdala, acc) are di erent from c = 2; : : : ; 5? This is a type of 
#     scientic question one could infer from the data. Whether or not there is a di erence 
#     between brain structure and political view.
# 

# MA ref: https://stackoverflow.com/questions/70655774/vectorization-of-numpy-matrix-that-contains-pdf-of-multiple-gaussians-and-multip

# (a) (5 points) Form the 1-dimensional histogram and KDE to estimate the distributions
# of amygdala and acc, respectively. For this question, you can ignore the variable 
# orientation. Decide on a suitable number of bins so you can see the shape of the 
# distribution clearly. Set an appropriate kernel bandwidth h > 0.
# 
# 
# https://stackoverflow.com/questions/53823349/how-can-you-create-a-kde-from-histogram-values-only

# In[1]:


import csv
import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats as st  #joint distribution #gaussian_kde
from scipy.stats import norm, permutation_test
import scipy.sparse.linalg as ll
from sklearn import preprocessing # data frame
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import seaborn as sns


# In[2]:


df = pd.read_csv('n90pol.csv')  #.to_numpy()
scaler = MinMaxScaler()
df[['amygdala', 'acc']] = scaler.fit_transform(df[['amygdala', 'acc']])
y = df.loc[:, ('orientation')]
y1 = df.loc[:, ('amygdala')]
y2 = df.loc[:, ('acc')]
X1 = df.loc[:, ('acc','amygdala')]
X = df.copy() 
df.info()
df.head(2)


# ### (a) (5 points) Form the 1-dimensional histogram and KDE to estimate the distributions of amygdala and acc, respectively. For this question, you can ignore the variable orientation. Decide on a suitable number of bins so you can see the shape of the distribution clearly. Set an appropriate kernel bandwidth h > 0.
# 

# In[3]:


def histogram_KDE_processor (df,feature,b,bw):
    return df[feature].plot.hist(bins=b), X1[feature].plot.kde(bw_method=bw) 

cd22 = histogram_KDE_processor(X1,'amygdala', 12, .2)  #bins 12, bw .2


# In[4]:


cd23 = histogram_KDE_processor(X1,'acc', 12, .2)


# ### (b) (5 points) Form 2-dimensional histogram for the pairs of variables (amygdala, acc). Decide on a suitable number of bins so you can see the shape of the distribution clearly.

# In[5]:


#Code Source: Provided by Instructor
data = pd.read_csv('n90pol.csv').to_numpy()
y = data[:,-1]
data2 = data[:,:2]
pdata = preprocessing.scale(data2)
m, n = pdata.shape

# for 2 dimensional data
#ax.scatter3D(x, y, z, c = y_train_new, marker = 'o', alpha=0.8, s=100, cmap='tab10', edgecolor='k')
min_data = pdata.min(0)
max_data = pdata.max(0)
nbin = 30        # you can change the number of bins in each dimension
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(pdata[:,0], pdata[:,1], bins=nbin)
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)
dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz)  #cmap='tab10'
ax.set_xlabel('amygdala')
ax.set_ylabel('acc')
#ax.set_zlabel('PDF')

#kernel density estimator
# create an evaluation grid
gridno = 40
inc1 = (max_data[0]-min_data[0])/gridno
inc2 = (max_data[1]-min_data[1])/gridno
gridx, gridy = np.meshgrid( np.arange(min_data[0], max_data[0]+inc1,inc1), np.arange(min_data[1], max_data[1]+inc2,inc2) )
gridall = [gridx.flatten(order = 'F'), gridy.flatten(order = 'F')]
gridall = (np.asarray(gridall)).T
gridallno, nn= gridall.shape
norm_pdata = (np.power(pdata, 2)).sum(axis=1)
norm_gridall = (np.power(gridall, 2)).sum(axis=1)
cross = np.dot(pdata,gridall.T)

# compute squared distance between each data point and the grid point;
#dist2 = np.matlib.repmat(norm_pdata, 1, gridallno)
dist2 = np.repeat(norm_pdata, repeats =gridallno).reshape((len(norm_pdata), gridallno))+np.tile(norm_gridall, m).reshape((len(norm_pdata), gridallno)) - 2* cross

#choose kernel bandwidth 1; please also experiment with other bandwidth;
bandwidth = .08

#evaluate the kernel function value for each training data point and grid
kernelvalue = np.exp(-dist2)

#sum over the training data point to the density value on the grid points;
# here I dropped the normalization factor in front of the kernel function,
# and you can add it back. It is just a constant scaling;
mkde = sum(kernelvalue) / m

from matplotlib import cm

#reshape back to grid;
mkde = ((mkde.T).reshape(gridno+1, gridno+1)).T
fig = plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.plot_surface(gridx, gridy, mkde, cmap=cm.coolwarm)

# Add a color bar which maps values to colors.
ax.set_xlabel('amygdala')
ax.set_ylabel('acc')
#ax.set_zlabel('PDF')

plt.show()


# ### (c) (20 points) Use kernel-density-estimation (KDE) to estimate the 2-dimensional density function of (amygdala, acc) (this means for this question, you can ignore the variable orientation). Set an appropriate kernel bandwidth h > 0. Please show the two-dimensional KDE (e.g., two-dimensional heat-map, two-dimensional contour plot, etc.) See down.
# 
# ### Please explain what you have observed: is the distribution  unimodal or bi-modal? Are there any outliers? Are the two variables (amygdala, acc) likely to be independent or not? (NOTE:  It actually involves prerequisite knowledge. From traditional probability and statistics, how do you show that two random variables are independent? Once you can answer that, then you need to visually represent that rule somehow and test for independence. )  Please support your argument with reasonable investigations.

# The first component is the definition: Two variables are independent when the distribution of one does not depend on the the other. In practice, we can check this by using the conditional distribution. If the probabilities of one variable remains fixed, regardless of whether we condition on another variable, then the two variables are independent. Otherwise, they are not.
# 
# The second component involves sampling: We do not often have access to the probabilities that generate a variable. We have access only to data attained through sampling. This means that there is some room for error. The observed conditional frequencies do not have to be exactly equal for the data to be independent: they need only be roughly equal. We can quantify what it means to be roughly equal, but, here, we’ll use a less rigorous, graphical approach.
# 
# 
# 
# "Estimate mutual information for a discrete target variable.
# 
# Mutual information (MI) [1] between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
# 
# The function relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances as described in [2] and [3]. Both methods are based on the idea originally proposed in [4].
# 
# It can be used for univariate features selection, read more in the User Guide."
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression

# ### C., (cont.) Are the two variables (amygdala, acc) likely to be independent or not? (NOTE:  It actually involves prerequisite knowledge. From traditional probability and statistics, how do you show that two random variables are independent? Once you can answer that, then you need to visually represent that rule somehow and test for independence. )  Please support your argument with reasonable investigations.
# 
#     -Probability 
# 
#     -Investigations or tests for independence:
#         i    Mutual info 
#         ii.  premutation Better for small data sets. (Appraopriate non-parametric)
#         iii. spearman test & p-value (Appraopriate non-parametric)
#    

# ### Probability, from traditional probability and statistics
# 
# 
# "Two random variables are independent if knowing the value of one does not change the probability of the other. This means that if X and Y are independent, we can write:
# 
#                     P(Y=y|X=x)=P(Y=y) 
#     
# for all x,y.
# 
# "The probability that a discrete random variable X takes on a particular value x i.e. P(X=x) is denoted by f(x) and is called the probability mass function (p.m.f.).discrete: probability mass function (p.m.f.). It is referred to as the probability density function (p.d.f.) for continuous random variables."
# https://towardsdatascience.com/independence-covariance-and-correlation-between-two-random-variables-197022116f93

# 
# 
# 
# 
# "Let $(X,Y)$ be a pair of random variables with values over the space ${\mathcal {X}}\times {\mathcal {Y}}$. If their joint distribution is ${\displaystyle P_{(X,Y)}}$ and the marginal distributions are  P_X and 
# ${\displaystyle P_{Y}}$, the mutual information is defined as 
# 
# $$I(X;Y) = D_{KL}(P_{(X,Y)}||\displaystyle P_{X}\otimes P_{Y})$$
# 
# where $D_{{{\mathrm  {KL}}}}$ is the Kullback–Leibler divergence, and ${\displaystyle P_{X}\otimes P_{Y}}$ is the outer product distribution which assigns probability ${\displaystyle P_{X}(x)\cdot P_{Y}(y)}$ to each $(x,y)$.
# Notice, as per property of the Kullback–Leibler divergence, that $I(X;Y)$ is equal to zero precisely when the joint distribution coincides with the product of the marginals, i.e. when $X and Y$ are independent (and hence observing Y tells you nothing about X). $I(X;Y)$ is non-negative, it is a measure of the price for encoding  $(X,Y)$ as a pair of independent random variables when in reality they are not."
# https://en.wikipedia.org/wiki/Mutual_information
# 
# 
# 
# pdf(x1, x2, ..., xn) = f(x1) * f(x2) * ... * f(xn)
# pdf(x, y) = f(x) * f(y)
# https://www.google.com/search?q=calculate+joint+pdf+for+numpy+matrix+continuous&sca_esv=568551326&sxsrf=AM9HkKlDQYpuQnUdea_YcIH1CAKM9rRI0A%3A1695747102193&ei=HgwTZeulC9myqtsP-f6KkAw&ved=0ahUKEwjrrJ6d3siBAxVZmWoFHXm_AsIQ4dUDCBA&uact=5&oq=calculate+joint+pdf+for+numpy+matrix+continuous&gs_lp=Egxnd3Mtd2l6LXNlcnAiL2NhbGN1bGF0ZSBqb2ludCBwZGYgZm9yIG51bXB5IG1hdHJpeCBjb250aW51b3VzMgQQIxgnSJgdUABYAHAAeAGQAQCYAXSgAcABqgEDMS4xuAEDyAEA-AEB4gMEGAAgQYgGAQ&sclient=gws-wiz-serp

# ###  C.i  mutual info:  Mutual information  returns zero for independent variables and higher values the more dependence there is between the variables.
# 
# 
# The computation of the mutual information function is based on the application of nonparametric techniques, based on entropy estimation from  k-nearest neighbors distances. This methodology is drawn from the foundational concepts initially introduced by L. F. Kozachenko, & N. N. Leonenko in 1987 [1].  Mutual information (MI) serves as a mathematical measure quantifying the dependency between two variables. This measure assumes non-negative values. MI attains a value of zero if the two variables are independent, with larger MI values indicative of greater interdependency.
# 
# Sklearn.feature_selection "mutual_info_regression " estimate mutual information for a continuous target variable.  See "mutual_info_classif" for discrete target variable.[2]
# 
# 1. L. F. Kozachenko, N. N. Leonenko, “Sample Estimate of the Entropy of a Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16
# 2. https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html

# In[6]:


from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)
data = pd.read_csv('n90pol.csv').to_numpy()
scaler = MinMaxScaler()
data[:, :2] = scaler.fit_transform(data[:, :2])
data2 = data[:,:2]
X = data2
acc = data2[:,1]
mi = mutual_info_regression(X, acc)
mi /= np.max(mi)
plt.figure(figsize=(15, 5))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.scatter(X[:, i], acc, edgecolor="blue", s=20)
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
    if i == 0:
        plt.ylabel("$acc$", fontsize=16)
    plt.title(" MI={:.2f}".format( mi[i]), fontsize=16)
plt.show()
#Code source:  https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py


# ### C.  ii.  Permutation:  A p-value greater than 0.05 means that deviation from the null hypothesis is not statistically significant
# 

# In[7]:


data = pd.read_csv('n90pol.csv').to_numpy()
scaler = MinMaxScaler()
data[:, :2] = scaler.fit_transform(data[:, :2])

Y = data[:,-1]
x = data[:,0]
y = data[:,1]

data2 = data[:,:2]
amygdala = data2[:,0]
acc = data2[:,1]
# Permutation on All data
def statistic(x, y, axis):  #statistic output
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


x = data[:,0]
y = data[:,1]
statistic(x, y, 0)
# test statistic is negative, the true mean of the distribution underlying x is less than 
#that of the distribution underlying y.
# because our statistic is vectorized, we pass `vectorized=True`
# `n_resamples=np.inf` indicates that an exact test is to be performed
res = permutation_test((x, y), statistic, vectorized=True, n_resamples= 100000, alternative='less')
df0 = pd.DataFrame(np.array([[res.statistic, res.pvalue]]), columns=['d_mean', 'pvalue'], index = [1])

def statistic(x, y, axis):  #statistic output
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def permutation_processor (data, n):
    desired_values = np.array([n]) #n is wat u are seectin for
    mask = np.isin(element = data[:,2],test_elements = desired_values)
    data_w = data[mask]
    x = data_w[:,0]
    y = data_w[:,1]
    statistic(x, y, 0)
    # test statistic is negative, the true mean of the distribution underlying x is less than that of the distribution underlying y.
    # because our statistic is vectorized, we pass `vectorized=True`
    # `n_resamples=np.inf` indicates that an exact test is to be performed
    res = permutation_test((x, y), statistic, vectorized=True, n_resamples= 100000, alternative='less')
    return res.statistic, res.pvalue


values = [2, 3, 4, 5]
sp = [permutation_processor(data,x) for x in values]
df = pd.DataFrame(sp, columns=['d_mean', 'pvalue'], index = [2, 3, 4, 5])

df_c = pd.concat([df0,df])
#df_c.T

df_c.plot(kind='line', title='Conditional Permutation of Amygdala & Acc',
               ylabel='Value', xlabel='Political Orientation (1 = All)', color = ('b', 'g'),grid = True, figsize=(6, 4))  #linewidth=0.7
plt.legend(title = "Measure")
plt.xticks(rotation=0)  #amygdala = data2[:,0]  Orientation
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
df_c.T


# ### C.i. spearman test
# 
# Perfect: If the value is near ± 1, then it said to be a perfect correlation: as one variable increases, the other variable tends to also increase (if positive) or decrease (if negative).
# High degree: If the coefficient value lies between ± 0.50 and ± 1, then it is said to be a strong correlation.
# Moderate degree: If the value lies between ± 0.30 and ± 0.49, then it is said to be a medium correlation.
# Low degree: When the value lies below + .29, then it is said to be a small correlation.
# No correlation: When the value is zero.

# In[8]:


from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('n90pol.csv').to_numpy()
scaler = MinMaxScaler()
data[:, :2] = scaler.fit_transform(data[:, :2])
y = data[:,-1]
data2 = data[:,:2]
amygdala = data2[:,0]
acc = data2[:,1]
#amygdala, acc


# In[9]:


dfCorr6 = X1.corr(method = 'spearman')
dfCorr6.style.background_gradient(cmap='coolwarm').set_precision(2)  


# In[10]:


res = st.spearmanr(amygdala, acc)
res_df_spear = pd.DataFrame([[res.statistic, res.pvalue]], columns=['stats', 'pvalue'], index = [1])

#res.statistic, res.pvalue
#Spearman  correlation coefficient (if only 2 variables are given as parameters). 
#Correlation coefficient is square with length equal to total number of variables (columns or rows) 
#in a and b combined.
#SignificanceResult(statistic=-0.1007577410393683, pvalue=0.34470088504373364)

#he p-value for a hypothesis test whose null hypothesis is that two samples have no ordinal correlation

data = pd.read_csv('n90pol.csv').to_numpy()
scaler = MinMaxScaler()
data[:, :2] = scaler.fit_transform(data[:, :2])
y = data[:,-1]
data2 = data[:,:2]
amygdala = data2[:,0]
acc = data2[:,1]

#Condition on opinion = 2,3,4,5
def conditional_processor_spearmanr (data, n):
    desired_values = np.array([n])
    mask = np.isin(element = data[:,2],test_elements = desired_values)
    data_2 = data[mask]
    amygdala = data2[:,0]
    acc = data2[:,1]
    res = st.spearmanr(amygdala, acc)
    return res
 

values = [2, 3, 4, 5]
spear = [conditional_processor_spearmanr(data,x) for x in values]
df_spear = pd.DataFrame(spear, columns=['stats', 'pvalue'], index = [2, 3, 4, 5])

df_sp = pd.concat([res_df_spear,df_spear])

df_sp.plot(kind='line', title='Conditional Spearmans of Amygdala & Acc',
               ylabel='Value', xlabel='Political Orientation (1 = All)', color = ('b', 'g'),grid = True, figsize=(6, 4))  #linewidth=0.7
plt.legend(title = "Measure")
plt.xticks(rotation=0)  #amygdala = data2[:,0]  Orientation
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
df_sp.T


# ### (d) (10 points) We will consider the variable orientation and consider conditional 
# distributions. Please plot the estimated conditional distribution of amygdala 
# conditioning on political orientation: p(amygdala j orientation = c), c = 2; : : : ; 5, 
# using KDE. Set an appropriate kernel bandwidth h > 0. Do the same for the volume of
# the acc: plot
# p(accjorientation = c), c = 2; : : : ; 5 using KDE. (Note that the conditional 
# distribution can be understood as  tting a distribution for the data with the same
# orientation. Thus you should plot 8 one-dimensional distribution functions in total 
# for this question.)Now please explain based on the results, can you infer that the
# conditional distribution of amygdala and acc, respectively, are di erent from c = 2; :
# : : ; 5? This is a type of
# scienti c question one could infer from the data: Whether or not there is a di erence
# between brain structure and political view.
# 
# 
# Now please also  ll out the conditional sample mean for the two variables:
# 
# 
# 
# Remark: As you can see this exercise, you can extract so much more information from
# density estimation than simple summary statistics (e.g., the sample mean) in terms of
# explorable data analysis.

# In[11]:


df = pd.read_csv('n90pol.csv')  #.to_numpy()
scaler = MinMaxScaler()
df[['amygdala', 'acc']] = scaler.fit_transform(df[['amygdala', 'acc']])
y = df.loc[:, ('orientation')]
y1 = df.loc[:, ('amygdala')]
y2 = df.loc[:, ('acc')]
X1 = df.loc[:, ('acc','amygdala')]
X = df.copy() 
df.info()
df.head(2)


# w = sns.displot(data=df, x="amygdala", col="orientation", bins=12, kde=True, height=3, aspect=.7)
# ww = sns.displot(data=df, x="acc", col="orientation", bins=12, kde=True, height=3, aspect=.7, color='g')
# 
# #https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot

# In[12]:


df_groups_w = df.groupby(['orientation']).agg(['mean'])
df_groups_w.T


# In[13]:


df_groups_ww = df.groupby(['orientation']).agg(['mean'])
df_groups_ww.plot(kind='line', title='Mean by Orientation & Feature',
               ylabel='Value', xlabel='Orientation', color = ('b', 'g'),figsize=(6, 4))  #linewidth=0.7
plt.legend(title = "Mean by Feature")
plt.xticks(rotation=0)
df_groups_w.T


# In[14]:


#figure(figsize=(width, height))
def conditional_distributions_processor (feature, o,b,bw):
    data_2=df[(df['orientation']== o)]
    return data_2[feature].plot.hist(bins=b), data_2[feature].plot.kde(bw_method=bw,color = 'orange' ) 


fig = plt.figure(figsize=(10, 3))
gs = fig.add_gridspec(1, 4, hspace=0, wspace=0)
(ax1, ax2,ax3, ax4) = gs.subplots(sharex='col', sharey='row')
ax1 = plt.subplot(1,4,1).set_title('Orientation 2')
conditional_distributions_processor('amygdala', 2, 12, .2)
ax2 = plt.subplot(1,4,2).set_title('Orientation 3')
conditional_distributions_processor('amygdala', 3, 12, .18)#(ax=plt.gca())
ax3 = plt.subplot(1,4,3).set_title('Orientation 4')
conditional_distributions_processor('amygdala', 4, 12, .18)#(ax=plt.gca())
ax4 = plt.subplot(1,4,4).set_title('Orientation 5')
conditional_distributions_processor('amygdala', 5, 12, .18)#(ax=plt.gca())
plt.tight_layout()
#amygdala


# def conditional_distributions_processor (feature, o,b,bw):
#     data_2=df[(df['orientation']== o)]
#     return data_2[feature].plot.hist(bins=b), data_2[feature].plot.kde(bw_method=bw) 
# 
# plt.figure(1)
# plt.subplot(2,2,1).set_title('Orientation 2')
# conditional_distributions_processor('amygdala', 2, 12, .2)
# plt.subplot(2,2,2).set_title('Orientation 3')
# conditional_distributions_processor('amygdala', 3, 12, .18)#(ax=plt.gca())
# plt.subplot(2,2,3).set_title('Orientation 4')
# conditional_distributions_processor('amygdala', 4, 12, .18)#(ax=plt.gca())
# plt.subplot(2,2,4).set_title('Orientation 5')
# conditional_distributions_processor('amygdala', 5, 12, .18)#(ax=plt.gca())
# #plt.set_title('title')
# plt.tight_layout()
# #amygdala # y is density & frequency

# In[15]:


sns.kdeplot(
   data=df, x="amygdala", hue="orientation",
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)


# In[16]:


#figure(figsize=(width, height))
def conditional_distributions_processor (feature, o,b,bw):
    data_2=df[(df['orientation']== o)]
    return data_2[feature].plot.hist(bins=b, color = 'r'), data_2[feature].plot.kde(bw_method=bw,color = 'orange' ) 


fig = plt.figure(figsize=(10, 3))
gs = fig.add_gridspec(1, 4, hspace=0, wspace=0)
(ax1, ax2,ax3, ax4) = gs.subplots(sharex='col', sharey='row')
ax1 = plt.subplot(1,4,1).set_title('Orientation 2')
conditional_distributions_processor('acc', 2, 12, .2)
ax2 = plt.subplot(1,4,2).set_title('Orientation 3')
conditional_distributions_processor('acc', 3, 12, .23)#(ax=plt.gca())
ax3 = plt.subplot(1,4,3).set_title('Orientation 4')
conditional_distributions_processor('acc', 4, 12, .28)#(ax=plt.gca())
ax4 = plt.subplot(1,4,4).set_title('Orientation 5')
conditional_distributions_processor('acc', 5, 12, .18)#(ax=plt.gca())
plt.tight_layout()

#acc


# In[ ]:


#sns.jointplot(data=df, x="amygdala", y="acc", hue="orientation", palette=[ '#3ba3ec', '#ffd92f', '#fab0e4', '#f77189'])  #crest
#https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette


# In[ ]:


#print(sns.color_palette("Set2").as_hex())  #husl  pastel6
#b: '#8da0cb'  y: '#ffd92f'


# In[ ]:


#sns.color_palette("pastel")
#print(sns.color_palette("pastel").as_hex())  #husl  pastel6
#pin '#fab0e4'


# In[ ]:


#print(sns.color_palette("husl").as_hex())  #husl  pastel6
#r: '#f77189'    : '#36ada4'


# In[ ]:


#'#8da0cb', '#ffd92f',  '#36ada4', '#f77189'


# ### (e) (5 points) Again we will consider the variable orientation. We will estimate the
# conditional joint distribution of the volume of the amygdala and acc, conditioning on
# a function of political orientation: p(amygdala; accjorientation = c), c = 2; : : : ;
# 5. You will use two-dimensional KDE to achieve the goal; et an appropriate kernel
# band-width h > 0. Please show the two-dimensional KDE (e.g., two-dimensional heat-map,
# two-dimensional contour plot, etc.).  #KDE of amygdala and acc w/ Political orientation = n
# 
# Please explain based on the results, can you infer that the conditional distribution of
# two variables (amygdala, acc) are di erent from c = 2; : : : ; 5? This is a type of 
# scientic question one could infer from the data. Whether or not there is a di erence 
# between brain structure and political view.

# In[18]:


data = pd.read_csv('n90pol.csv').to_numpy()
scaler = MinMaxScaler()
data[:, :2] = scaler.fit_transform(data[:, :2])
y = data[:,-1]
data2 = data[:,:2]
amygdala = data2[:,0]
acc = data2[:,1]

#Condition on opinion = 2,3,4,5
def conditional_processor (data, n):
    desired_values = np.array([n])
    mask = np.isin(element = data[:,2],test_elements = desired_values)
    data_2 = data[mask]
    return data_2

def kde_3d_processor (data):
    x = data[:,0] #amygdala
    y = data[:,1]  #acc
    # grid borders
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    # grid 
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    #kde 
    p = np.vstack([xx.ravel(), yy.ravel()])
    v = np.vstack([x, y])
    k = st.gaussian_kde(v,.3) ########## < data, bw_methodstr,may try various #s
    f = np.reshape(k(p).T, xx.shape)
    #3D KDE plots
    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('amygdala')
    ax.set_ylabel('acc')
    ax.set_zlabel('PDF')
    fig.colorbar(surf, shrink=0.5, aspect=5) #  color indicating  PDF
    return ax.view_init(60, 35)

 


# In[19]:


#Compare w/ Non-conditional:
kde_3d_processor (data)
plt.title('Orientation All ',fontsize=10)

data_2 = conditional_processor (data, 2)
kde_3d_processor (data_2)
plt.title('Orientation 2',fontsize=10)

data_3 = conditional_processor (data, 3)
kde_3d_processor (data_3)
plt.title(' Orientation 3',fontsize=10)

data_4 = conditional_processor (data, 4)
kde_3d_processor (data_4)
plt.title(' Orientation 4',fontsize=10)

data_5 = conditional_processor (data, 5)
kde_3d_processor (data_5)
plt.title(" Orientation 5", fontsize=10)
#probability density function


# In[22]:


table2 = df.groupby(['orientation'])
table2


# In[23]:


table = df.groupby(['orientation']).agg(['skew','var','mean'])
table


# In[24]:


table.T


# In[25]:


ax2 = df.plot.scatter(x='amygdala', y='acc',c='orientation',colormap='rainbow') #hsv  autumn  Set2 paired


# In[26]:


ax1 = df.plot(kind='scatter', x='acc', y='orientation',label='acc', color='r', figsize=(6, 4))   #mediumseagreen , Skyblue 
ax2 = df.plot(kind='scatter', x='amygdala', y='orientation', label='amygdala',color='cornflowerblue', ax=ax1)    
#ax3 = df.plot(kind='scatter', x='e', y='f', color='b', ax=ax1)
ax.set_xlabel("acc & amygdala")
ax.set_ylabel("orientation")
print(ax1 == ax2)  


# In[27]:


df.plot.scatter(x='acc', y='orientation',figsize=(3, 2), c='r')  #'mediumseagreen'
df.plot.scatter(x='amygdala', y='orientation',figsize=(3, 2), c='cornflowerblue')
classes = ['Acc', 'Amygdala']
#plt.legend(labels=classes)
plt.show()


# In[28]:


def conditional_distributions_processor (feature, o,b,bw):
    data_2=df[(df['orientation']== o)]
    return sns.kdeplot(data=data_2, x="amygdala", y="acc", hue="orientation", palette = "coolwarm", fill=True)   #levels=5, thresh=.2, palette="crest"


# In[29]:


def conditional_kdeplot_scatterplot_processor (o,l,bw):
    data_2=df[(df['orientation']== o)]
    return sns.kdeplot(data=data_2, x="amygdala", y="acc",  fill=True, levels=l, thresh=bw), sns.scatterplot(data=data_2, x="amygdala", y="acc") 

plt.figure(1)
plt.subplot(2,2,1).set_title('Orientation 2')
conditional_kdeplot_scatterplot_processor(2, 15, .01)
plt.subplot(2,2,2).set_title('Orientation 3')
conditional_kdeplot_scatterplot_processor( 3, 5, .07)#(ax=plt.gca())
plt.subplot(2,2,3).set_title('Orientation 4')
conditional_kdeplot_scatterplot_processor(4, 5, .07)#(ax=plt.gca())
plt.subplot(2,2,4).set_title('Orientation 5')
conditional_kdeplot_scatterplot_processor( 5, 5, .07)#(ax=plt.gca())
plt.tight_layout()


# In[30]:


sns.kdeplot(data=df, x="amygdala", y="acc", hue="orientation", palette="coolwarm",  levels=5, thresh=.1,)
#sns.scatterplot(data=df, x="total_bill", y="tip", hue="time")
sns.scatterplot(data=df, x="amygdala", y="acc", hue="orientation", palette="coolwarm" )  #,  levels=5, thresh=.1,)
#sns.scatterplot(data=tips, x="total_bill", y="tip", hue="size")

