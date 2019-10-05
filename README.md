****

# Recurrent Tensor Factorization 

This is the implementation of our paper:

Yiwen Zhang, Chunhui Yin, Zhihui Lu*, Dengcheng Yan, Meikang Qiu, Qifeng Tang. [Recurrent Tensor Factorization for time-aware Service Recommendation](https://doi.org/10.1016/j.asoc.2019.105762), Applied Soft Computing 85 (2019) 105762. (SCI)

Author: Chun-hui Yin

Affiliate: [Big Data and Cloud Service Lab](http://bigdata.ahu.edu.cn), Anhui University

Last updated: 2019/10/05

**Please cite our paper if you use our codes. Thanks!** 

# Environment Requirement

This code can be run at following requirement but not limit to:
- python = 3.6.6
- tensorflow-gpu = 1.7.0
- keras = 2.0.9
- pandas = 0.23.4
- numpy = 1.14.0
- scikit-learn = 0.21
- other installation dependencies required above

# Example of Usage

&gt;&gt;&gt;python RTF.py

&gt;&gt;&gt;python GTF.py

&gt;&gt;&gt;python PGRU.py

# Dataset

- To simulate the real-world situation, we sparse the original matrix at 4 densities and generate instances for training
- Here we provide the preprocessed real-world dataset WS-Dream (dataset#2)
- The original WS-DREAM dataset can be downloaded at [InplusLab](http://inpluslab.com/wsdream/)

# Note

- Experiments can be run on multi-core CPUs at 6 densities by turning on parallel mode
