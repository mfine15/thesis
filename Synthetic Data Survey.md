# Synthetic Data Survey

### Hardt Ligett and McSherry 2010 

Here, the authors combine the multiplicative weights approach to privately compute a data distribution that closely mirrors the original, with improved accuracy compared to prior work. A synthetic database can be created by repeatedly sampling from this database. 

We define a counting query takes a function $f: X \to [0,1]$ and returns $f(D) = E_{x \sim D}f(x)$. Informally, given a dataset  $D \subset X$ of size $|D| = n$ and a set of counting queries Q, this paper presents an algorithm for computing an $\epsilon$DP distribution $D^*$ so that with high probability $D^*$ satisfies
$$
\forall f \in Q: ~~~|f(D) - f(D^*)| \leq O\left(\frac{\log|X| \log|Q|}{\epsilon n}\right)^{1/3}
$$
Relaxing to $\epsilon, \delta$ - DP, allows us to improve this bound to 
$$
\forall f \in Q: ~~~|f(D) - f(D^*)| \leq O\left(\frac{\sqrt{\log|X| \log(1/\delta)} \log|Q|}{\epsilon n}\right)^{1/2}
$$
Has runtime
$$
O(n|Q| + T\cdot|D|\cdot|Q|)
$$
where n is the size of the synthetic dataset, Q the query universe, D the data universe, and T the number of steps. 

##### Locality

To make this local, need a 

### DualQuery

Because the record is computed non privately, this can trivially be modified to be locally DP by simply sneding the chosen query to the clients, and aggregating their noisy report of the query error.	

Runtime is linear with size of data universe (and thus exponential with dimension of data). This is impossible to improve in the worst case. However, runtime is also linear with size of *query* universe (again, exponential with dimension of queries) — can we improve this? 

### BLR

### Hardness of Non-Interactive Differential Privacy from One-Way Functions

Shows that there is no general purpose, non-interactive DP algorithm for answering queries when X and Q are exponential large, and where the dataset is poly() dimensional

### How Generative Adversarial Networks and Their Variants Work: An Overview

* f-divergence lets us use an convex function to judge the accuracy of the generated database — maybe this lets us make specialized Gans
* Alternatively, integral probability metric might let us do the same thing

![image-20190614171114631](/Users/Michael/Library/Application Support/typora-user-images/image-20190614171114631.png)

![image-20190614171607234](/Users/Michael/Library/Application Support/typora-user-images/image-20190614171607234.png)

![image-20190614171620261](/Users/Michael/Library/Application Support/typora-user-images/image-20190614171620261.png)