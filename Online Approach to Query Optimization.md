# Online Approach to Query Optimization

(Grnarova et al 2017) use a similar approach to (Oliehook et al) combined with an online learning approach. They describe a novel training method *Chekhov GAN* that provably coverges to an equlibrium when the discriminator is a one layer network (and an arbritrary generator). Not clear how to extend this to realistic, useful discriminators. Note that this also relies on having an oracle that to find the minimum over a sum of generative networks -- making it very similar to oraclequery.

However, a later paper by (Agarwal, Gonen, Hazan 2019) *might* suggest an approach that doesn't rely on the concavity of the discriminator.

Also c.f. "Private Learning Implies Online Learning: An Efficient Reduction"



### Background: Follow the Regularized Leader

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

##### 

### Algorithm

Defn: A game $M$ is **semi-concave** if for any fixed strategy of one player, the game is concave with respect to the play of the other player

**Theorem 1:** Let $K_2$ be a convex set, $M$ be a semi-concave zero-sum game that is $L-$Lipshcitz. Upon running CHEKHOV GAN algorithm for $T$ steps, using the instantiation of FTRL $A_1$, $A_2$ for the generator and discriminator respectively, then it outputs mixed strategies that are $\epsilon$-mixed nash equlibrium, for $\epsilon = O(1/\sqrt{T})$. 
$$
(A_1)~~~ u_t \leftarrow \arg\min_{u\in K_1} \sum _{\tau=0}^{t-1} f_\tau(u)\\
(A_2)~~~ v_t \leftarrow \arg\max_{u\in K_2} \sum _{\tau=0}^{t-1} \nabla g_\tau(v_\tau)^\intercal v - \frac{\sqrt{T}}{2\eta_0}||v||^2
$$

### Other Notes

Ala  "Two-Player Games for Efficient Non-Convex Constrained Optimization", robust optimization (min-maxing the worst case query error) can be transformed into constrained optimization with the addition of a slack variable

