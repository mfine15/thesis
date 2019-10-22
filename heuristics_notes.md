Is the $1/\delta$ runtime necessary? It's used to convert a non-robustly dp algo into a robustly dp algo by essentially ensuring that with probability $< 1/\delta$ the aggregated oracles won't fail.

What natural restrictions on the oracle failure modes can we use?

(Note that https://arxiv.org/pdf/1611.01688.pdf talks about the same problem). Specifically, the use of a translation matrix to ensure sensitivity might help



- Augmenting the dataset definitely fucks with optimality -- if one of the augmented queries happens to be minimal, (which it likely will be because there must be one that returns 0), that one has a high probability of being returned, incorrectly
- Also might fuck with privacy -- 





https://arxiv.org/pdf/1811.02002.pdf

https://arxiv.org/pdf/1806.07268.pdf