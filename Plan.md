Dear Salil,



I hope your summer is going well! I'm enjoying working at Two Sigma, and they actually brought in Adam Smith to give a talk on differential privacy. He had some really interesting comments on the state of the field. 

I wanted to reach out because I've been doing a good amount of reading 



Local Differentially Private:

​	Three approaches, all generally taking the approach of randomized response, combined with sometimes working on another representation of the data. Obviously can't be more computationally efficient than non-local, and all of them are exponential in the size of the dataset (and possibly query set TOOD). However, they can't easily handle greater than 32 dimensions. 

​	Instead, it would be interesting to see if we could 1) apply MWEM/DualQuery to the local setting, and 2) apply GAN w/ a federated learning approach. 

​	

* Combine DualQuery w/ some sort of PCA/PrivBayes
* How good is marginal release under LDP
* How useful are just marginals
* How often do GAN-style generative models preserve marginals
* Does a federated learning approach hold promise?
* (my job) what bounds can I prove? (w or w/o learning theory)