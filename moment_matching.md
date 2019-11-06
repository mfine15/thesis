# Query Release via Moment Matching

## Moment Matching Kernels

**Definition 1:** Let $\mathcal{F}$ be a class of functions and let $p$ and $q$ be the true data and fake data distribution respectively, and $X$ and $Y$ be finite observations drawn iid from $p,q$ respectively. We then define the maximum mean discrepancy and its empirical estimate as 
$$
MMD[\mathcal{F},p,q] := \sup_{f\in\mathcal{F}}\left( \mathbb{E}_{x\sim p}[f(x)] - \mathbb{E}_{y\sim q}[f(y)]\right)\\
MMD[\mathcal{F},X,Y] := \sup\left(_{f\in\mathcal{F}} \frac{1}{m}\sum_{i=1}^mf(x_i) -  \frac{1}{n}\sum_{i=1}^mf(y_i)\right)
$$
  We can define the squared maximum mean discrepancy (MMD) loss as

**[TODO can't make the jump from supremum to sum]**
$$
\begin{align*}
\mathcal{L}_{\mathrm{MMD}^{2}}&:=\left\|\frac{1}{N} \sum_{i=1}^{N} f\left(x_{i}\right)-\frac{1}{M} \sum_{j=1}^{M} f\left(y_{j}\right)\right\|^{2} \\
&=\frac{1}{N^{2}} \sum_{i=1}^{N} \sum_{i^{\prime}=1}^{N} f\left(x_{i}\right)^{\top} f\left(x_{i^{\prime}}\right)-\frac{2}{N M} \sum_{i=1}^{N} \sum_{j=1}^{M} f\left(x_{i}\right)^{\top} f\left(y_{j}\right) +\frac{1}{M^{2}} \sum_{j=1}^{M} \sum_{j^{n}} use the kernel trick to rewrite this in terms of the kernel $k$ where $k(a,b) = \langle f(a), f(b) \rangle$.
\end{align*}
$$
$$
\begin{aligned} \mathcal{L}_{\mathrm{MMD}^{2}} &=\frac{1}{N^{2}} \sum_{i=1}^{N} \sum_{i^{\prime}=1}^{N} k\left(x_{i}, x_{i^{\prime}}\right)-\frac{2}{N M} \sum_{i=1}^{N} \sum_{j=1}^{M} k\left(x_{i}, y_{j}\right) +\frac{1}{M^{2}} \sum_{j=1}^{M} \sum_{j^{\prime}=1}^{M} k\left(y_{j}, y_{j^{\prime}}\right) \end{aligned}
$$

## Boolean Kernels

Using the kernel trick, we can efficiently find the function $f \in \mathcal{F}$ that maximizes the difference :
$$
\mathbb{E}_{x\sim p}[f(x)] - \mathbb{E}_{y\sim q}[f(y)]
$$
Note that maximizing this discrepancy is exactly equivalent to maximizing the Wasserstein GAN objective.  As such, if we can find a kernel $k$ in a RKHS such that its feature map $\mathcal{F}$ describes a query class of interest, we can use $\mathcal{L}_{\mathrm{MMD}^{2}}$ as an efficient loss function for the *entire* class of queries. 

One such kernel is the kernel corresponding to all monotone monomials of length up to $d$, which we denote by $k_d$ [@KSW03]. [TODO define $\mathbb{B}$]
$$
k_d\left(\mathbf{x}, \mathbf{x}^{\prime}\right):=\left\langle f(\mathbf{x}), f\left(\mathbf{x}^{\prime}\right)\right\rangle_{K}=\sum_{\mathbf{i} \in \mathbb{B}^{n}_d} K_{\|\mathbf{i}\|}^{-1} \mathbf{x}^{\mathbf{i}} \mathbf{x}^{\mathbf{i}}
$$
Explicitly computing this would require summing $|\mathbb{B}_d^n| = O(n^d)$ terms. [TODO explain why, prove this], Thus,
$$
k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\sum_{j=0}^{d} e^{j}\left(\begin{array}{c}{\left\langle\mathbf{x}, \mathbf{x}^{\prime}\right\rangle} \\ { j}\end{array}\right)=(1+b)^{\left(\mathbf{x}, \mathbf{x}^{\prime}\right)}
$$
where $b$ determines the weight assigned to higher order polynomials. 

[TODO all subsets kernel]

**NOte:** A supremum of convex functions is convex


$$
\begin{array}{l}{\text { Lemma } 4 \text { Assume the condition in Lemma } 3 \text { for the existence of the mean embeddings } \mu_{p}, \mu_{q} \text { is }} \\ {\text { satisfed. Then }} \\ {\qquad \operatorname{MMD}^{2}[\mathcal{F}, p, q]=\left\|\mu_{p}-\mu_{q}\right\|_{\mathcal{H}}^{2}} \\ {\text { Proof }} \\ {\qquad \begin{aligned} \operatorname{MMD}^{2}[\mathcal{F}, p, q] &=\left[\sup _{\|f\|_{\mathcal{Y}_{\mathcal{F}} \leq 1}}\left(\mathbf{E}_{x}[f(x)]-\mathbf{E}_{y}[f(y)]\right)\right]^{2} \\ &=\left[\sup _{\|f\|_{\mathcal{B}} \leq 1}\left\langle\mu_{p}-\mu_{q}, f\right\rangle_{\mathcal{H}}\right]^{2} \\ &=\left\|\mu_{p}-\mu_{q}\right\|_{\mathcal{Y}^{\prime}}^{2} \end{aligned}}\end{array}
$$
