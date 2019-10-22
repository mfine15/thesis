- Generator is neural network trained by gradient descent
  - Can verify optimality with near zero loss on queries
- Discriminator is simply multiplicative weights over query set 
  - This lets us verify optimality for free
  - Maybe mixture of discriminators, some neural networks etc
  - Alternatively, train on empirical samples and boost
    - Probability of generalization => generation bounds, use differential privacy (nice little combo herer)
- Online algorithm trained like mixgan









