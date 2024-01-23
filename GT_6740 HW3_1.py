#!/usr/bin/env python
# coding: utf-8

# GT_6740 HW3_1c
# 
# 3. (5 points) For the EM algorithm for GMM, please show how to use the Bayes rule to
# drive $t_{i}^{k}$ in a closed-form expression
# 
# Bayes rule
# 
# $$P(z|x) = \frac{P(x|z)P(z)}{P(x)} = \frac{P(x,z)}{\sum_{z}^{'}P(x,z^{'})} $$
# 
# Prior:$p(z) = \pi_{z}$
# 
# Likelihood: $p(x|z) = N(x|\mu_{z},\sum_{z})$
# 
# Posterior: $p(z|x) = \frac {\pi_zN(X|\mu_{z},\sum_{z})}{\sum_{z^{'}}\pi_{z^{'}}N(X|\mu_{z^{'}},\sum_z{'}})$
# 
# 
# E-step: find the posterior distribution
# 
# $q(z^{1}, z^{2}, . . . . , z^{m})$:  posterior distribution of the latent variables in 𝑡-th iteration
# 
# 
# $$q(z^{1}, z^{2}, . . . . , z^{m})= \prod_{i=1}^{m} p(z^{i}|x^{i},\theta^{t})$$     
# 
# 
# 
# 
# For each data point $x^{i}$ , compute $p(z^{i} = 𝑘|x^{i})$  for each 𝑘
# 
# $$t_{𝑘}^{i} = p(z^{i} = 𝑘|x^{i},\theta^{t})= \frac{p(x^{i}|z^{i} = 𝑘)p(z^{i} = 𝑘)}{\sum_{𝑘^{'}-1..𝑘}p(z^{i} = 𝑘^{'},x^{i})}$$
# 
# $$ = \frac{\pi_{𝑘}N(x^{i}|\mu_{𝑘},\sum_{𝑘})}{\sum_{𝑘^{'}=1..𝑘}\pi_{𝑘^{'}}N(x^{i}|\mu_{𝑘^{'}},\sum_{𝑘^{'}})}$$
# 
# Source: OMSA_Week_7_Slides.pdf, Xie, Yao, Ph.D., Associate Professor, Georgia Institute of Technology 
# 
