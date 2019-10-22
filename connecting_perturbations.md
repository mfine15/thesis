# The value of perturbations in privacy and online learning

It's known that even with access to an optimization oracle, online learning is exponentially harder than statistical learning [@HK15]. However, In [@AGH18], by strengthening the oracle to allow it to minimize loss *subject to a linear perturbation*, they show that online and statistical learning models are equivalent. 

Simultaneously, [@NRW18] provided an oracle-efficient method for privately generating synthetic data for any query class that had a small *separator set*. Interestingly, we can show that in fact all query sets can be trivially modified to have a logarithmically sized separator set, if we allow arbritrary (TODO linear?) perturbations of the loss function provided to the oracle.

Interestingly then, allowing for perturbations of an oracle makes two disparate but related problems feasible -- both non-convex online learning, as well as *differentially private* learning. Does that suggest deeper connections?

## Background: Learning in non-convex games [@AGH18]

### Definitions

*Online learning* is the setting where in each round $t$ our algorithm $A$ picks a point $x_t$ from the data universe $\mathcal{X}$ and suffers loss according to an adversarially chosen loss function $l_t \in \mathcal{L}$.

The goal of the learner is to minimize *average regret*, the difference in average loss of the learner and the loss from the best fixed point $x^* \in \mathcal{X}$ in hindsight: 
$$
\text{Regret}_T(x) := \sum_{t=1}^T f_t(x_t) - f_t(x^*)
$$
Moreover, we give $A$ access to an offline perturbed optimization oracle $O$ whose input is a sequence of $k$ previous loss functions $(l_1...l_k) \in \mathcal{L}^k$ and a $d$-dimensional perturbation vector $\sigma$, and outputs the minimizer $\hat x$: 
$$
\hat x \in \arg\min_{x \in \mathcal{X}} \sum _{i=1}^k l_i(x) - <\sigma, x>
$$
(**Note: ** can we make this work when $x$ is only $\epsilon$-close to the true minimizer, and where $O$ only succeeds with probability $(1-\beta)$?)

### Value of linear perturbations

[TODO from [@HK15]]

### Background: Follow the Regularized Leader

FTRL is a broad approach to online learning. Given a sequence of online loss functions $l_1...l_{t-1}$, and an action set  $X$, how should we select $x_t$, the best response for the cost function $l_t$? One approach is select the one that performs best on the past cost functions
$$
x_t := \arg \min_{x\in X} \sum l_k(x)
$$
This approach is very exploitable, in essence because it "overfits" to past history. To fix this, we attempt to regularize. Given a "regularizer function" $R$ (such as $l_2$ norm), FTRL selects $x_t$ by
$$
x_t := \arg \min_{x\in X} R(x ) + \sum l_k(x)
$$
We can bound the regret after $T$ steps as 
$$
\text{Regret}_T(x) \leq \left(\sum_{t=1}^T l_t(x_t) - l_t(x_{t+1})\right) + R(x) - R(x_1)
$$
[TODO why doesn't $l_2$ regularization work for non-convex case pg 3]

### Non-convex FTPL

Instead, let FTPL be the algorithm where at each step, we draw a random vector $\sigma_t \sim Exp(\eta)^d$, and select $x_t$ by
$$
x_t \in \arg\min_{x \in \mathcal{X}} \sum _{i=1}^k l_i(x) - \sigma_t^Tx
$$
 However, we make analysis easier by assuming an oblivious adversary, s.t the sequence of loss functions is chosen in advance. This allows us to reuse the same $\sigma$ for all iterations. 

Using the lemma 4.1 from [@CL06], regret bounds for the oblivious case are asymptotically equivalent to non-oblivious adversaries.

#### Proof

#### Connection to Two-player games & GANs

### 