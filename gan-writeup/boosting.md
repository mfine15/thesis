QueryGAN requires that the GAN objective be convex w.r.t the parameters of the discriminator. While this is a limiting assumption, there is no requirement that $D$ be a single, single layer NN.



Instead, we can use boosting. 
$$
\hat D(x) := \sum_{i=1}^N w_iD_i(x)    \quad : \quad w \in \mathbb{R}^+
$$
As positive weighted sums of convex functions are also convex, our assumptions hold. 