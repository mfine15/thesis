# Towards Practical Private Data Release

## Introduction

​	In the standard model of differential privacy, analysts submit queries to a trusted curator, who returns a noisy answer to each in an online fashion. While simple, this paradigm is suboptimal in a number of ways. It requires analysts to profoundly change their workflow -- instead of being able to inspect, query, and manipulate data they have access to, they must now submit queries to a curator without ever being able to look at the data. Moreover, as [TODO cite] Dwork notes, this model gives up *analyst privacy* -- the curator is necessarily aware of each query the analyst makes of the data.

​	Ideally, the curator would release a *differentially private synthetic dataset* -- a data structure that, while differential private, "looks like" the true dataset. More formally, given a sensitive dataset $\mathcal{X} \in \mathbb{R}^{m\times n}$, we are looking for a differentially private sanitizer $M$ such that $\tilde{\mathcal{X}} = M(\mathcal{X}) $ approximates $\mathcal{X}$ with respect to a class of queries $Q$:
$$
\forall q \in Q : ~~ |q(\mathcal{X}) - q(\tilde{\mathcal{X}})| \leq \alpha
$$
for some constant $\alpha$. [TODO standardize notation]

[TODO lit review]

## DualQuery 

DualQuery views the problem of private data release as a zero-sum game between the query player and the data player. The value of the game is the difference between the query run on the data players move and the true dataset. In DualQuery's formulation, the query player uses a no-regret learning algorithm, while the data player finds a best response using an optimization algorithm. 

#### Nash Equilibrium analysis of Query accuracy

Assume there are two players, $G$ and $D$. $G$ can play any element in the data universe $\mathcal{U}$, and $D$ can play any element in the query universe $\mathcal{Q}$. Given a play $x \in \mathcal{U}$ and $q \in \mathcal{Q}$, let the payoff be
$$
A(x,q) := q(T) - q(x)
$$
where $T$ is the true dataset.

We can extend this pure definition to the mixed setting, where instead of fixed strategies each player plays *probability distributions* over their action sets. Let $\Delta\mathcal{U}$ and $\Delta\mathcal{Q}$ be the set of probability distributions over $\mathcal{U}$ and $\mathcal{Q}$. Now, given a strategy $u \in \Delta\mathcal{U}$ and $q \in\Delta \mathcal{Q}$ we define the payoff to be the expected value of an action drawn from each
$$
A(u,w) := \mathbb{E}_{x \sim u, q \sim w} A(x,q)
$$
By von neumann's minimax theorem, the mnimizing player can always force payoff at most $v_A$, while the maximizing player can always force payoff at least $v_A$. We thus call $v_A$ the value of the game. 

Thus, assuming that each query $q$ has a complementary query $\bar q = -q$, $v_A$ must equal 0. We then conclude that if we've reached an $\alpha-$approximate Nash Equilibrium, we've obtained our guarantee that
$$
\forall q \in Q : ~~ |q(\mathcal{T}) - q(x)| \leq \alpha
$$


## OracleQuery

OracleQuery generalizes the no-regret adversarial approach of DualQuery to support any heuristic optimization oracle, while reducing the oracle-runtime dependence to $\log |Q|$ rather than $|Q|$. Unfortunately, it has a few drawbacks

- Runtime depends linearly on $1/\delta$, which is infeasible when $\delta$ is cryptographically ($<10^{-100}$) small.
- Probabilistic optimization oracle must be able to certify that the solution it returns is optimal (though it is allowed to simply not return a result with small probability). While this works for certain optimization procedures, it is infeasible/impossible for procedures like gradient descent. 
- 

## Generative Adversarial Networks

​	In a parallel line of research, Generative Adversarial Networks (Goodfellow 2016) have shown great promise in generative realistic looking, high dimensionsional images. Quite similar to the DualQuery approach, a GAN is trained by pairing two deep neural networks, a generator and a discriminator. The generator aims to generate realistic samples, while the discriminator tries to distinguish between fake and real samples. 

​	Beyond standard problems with training GAN's (mode collapse etc), the primary issue with the GAN formulation is that so far we've not been able to make any theoretical guarantees about the worst case error (or even average case) of a query over GAN-generated dataset. This is extremely important for our scenario, where analysts would like to know with high probability any result they obtain on the synthetic dataset would approximately hold on the true dataset. This lack of theoretical guarantees actually stems from two related problems:

1. No guarantee that the converged solution is a global optimum (or global nash equilibrium)
2. No guarantee that even a global nash equilibria in the *parameter space* is really optimal in the *distribution space*. 
   1. **Possible solution:** If we only need to guarantee the approximation for a certain query class $Q$, if we can constrain the discriminator to resemble that query class Q, we no longer need guarantees for the rest of the distribution space. 

Let's reiterate: we don't need to prove convergence, just that convergence $\to$ accuracy

### How combine theoretical guarantees with distribution learning capabilities of GANs

In (Oliehook et al), they avoid the problem of local nash equlibria by introducing a resource bounded nash equilibirum (RB-NE), and provide an algorithm that is monotonically guaranteed to converge to a RB-NE. However, it's unclear how efficient this is, as it relies on a likely exponential-time search algorithm, parallel nash memory (PNM). 

In (Unterhiner et al 2018), they describe a physics inspired formulation Coulumb GAN that ensures an optimal nash equlibria. However, this only occurs under the strong assumption that "generator samples cannot move freely", which doesn't hold when training via gradient descent. Furthermore, their approach relies on a distance metric between data points, which is not typically required when training GANs.

### Online Approach to Query Optimization

(Grnarova et al 2017) use a similar approach to (Oliehook et al) combined with an online learning approach. They describe a novel training method *Chekhov GAN* that provably converges to an equlibrium when the discriminator is a one layer network (and an arbritrary generator). Not clear how to extend this to realistic, useful discriminators. Note that this also relies on having an oracle that to find the minimum over a sum of generative networks -- making it very similar to oraclequery.

##### Background: Follow the Regularized Leader

FTRL is a broad approach to online learning. Given a sequence of online cost functions $f_1...f_{t-1}$, and an action set  $K$, how should we select $x_t$, the best response for the cost function $f_t$? One approach is select the one that performs best on the past cost functions
$$
x_t := \arg \min_{x\in K} \sum f_k(x)
$$
This approach is very exploitable, in essence because it "overfits" to past history. To fix this, we attempt to regularize. Given a "regularizer function" $R$, FTRL selects $x_t$ by
$$
x_t := \arg \min_{x\in K} R(x ) + \sum f_k(x)
$$
We can bound the regret after $T$ steps as 
$$
\text{Regret}_T(x) \leq \left(\sum_{t=1}^T f_t(x_t) - f_t(x_{t+1})\right) + R(x) - R(x_1)
$$
where 
$$
\text{Regret}_T(x) := \sum_{t=1}^T f_t(x_t) - f_t(x)
$$

### Boosting for Queries/AdaBoost/AdaGan

#### Boosting and Differential Privacy

In *(Dwork Rothblum Vadhan 2010)*, the authors showed how to apply the concept of Boosting from learning theory to differential privacy. They show how to privately turn a weak synthetic database generator into a strong database geneator.

Let $M$ be a synopsis generator  that given a database $x \in \mathbb{X}^n$, for any $k$ queries sampled independently from some distribution $D$ over the query family $Q$,  $M$ outputs a synopsis that has max error $\lambda$ for a $(1/2+\eta)$ fraction of $Q$ with probability $\exp(-\kappa)$. 

This paper provides a procedure for taking $M$ and making it ($\lambda + \mu$)-accurate for all queries in $Q$ with probability at least $1-T\exp(-\kappa)$. $T$ is the number of rounds, $O(\log|Q|\eta^2)$.  