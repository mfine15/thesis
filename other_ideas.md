Challenge with all of these optimization oracle based approaches is that SGD can never provide (probabalistic) loss guarantees, let alone certify them. However, if we use SGD on the data set, there is a loss guarantee -- $0$, because that's the loss of the true dataset. 

Question is

1. Can we do this in a differentially private manner
2. Can we frame the optimization like this?

**NB:** DualQuery relies on the data player playing the ~worst~ performing record -- this relies on finding the best performing record



- Multiplicative weights on autoencoded data
- 