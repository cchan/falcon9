I spent a long time reading [this](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/) and feel like I understand Q-learning now.
I'm going to basically rewrite it below.

Q is the total reward. Q(s,a) takes the current state and action, and outputs the maximum total reward that you can achieve.

$$
Q(s,a) = r(s,a) + \gamma \max\limits_a' Q(s',a')
$$

The current maximum total reward is the sum of:
- the immediate reward $r(s,a)$ due to action $a$ on state $s$
- the highest possible reward over all actions $a'$ you could take from the next state $s'$
We discount the future reward by $\gamma$ (which might be equal to, say, 0.999) because it's helpful to be a bit greedy in the near term, and because this helps the recursion converge.

This is pretty much a dynamic programming trick to help figure out what the value of a given action is in a given state.

If we run this over the course of time $t$, and do some math regarding convergence, you can change the above into an update step to help the Q-table get better over time.

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[ R_{t+1} + \gamma \max\limits_a Q(S_{t+1},a) - Q(S_t,A_t) \right]
$$

where $\alpha$ is the learning rate.

With harder problems it starts being really difficult to store the whole Q-table, especially when things get continuous.
So DeepMind came up with Deep Q-learning, which considers Q to be an arbitrary function.
And guess what can model arbitrary functions! Neural networks. Awesome.

So now we have a supervised learning problem for Q, where we have a datapoint (S_t, A_t, S_{t+1}, R{t+1}), and expect
$Q(S_t, A_t)$ to be equal to $R_{t+1} + \gamma \max\limits_a Q(S_{t+1},a)$
and give it the usual gradient descent update if it's wrong.

The question here is how to do the $\max\limits_a$. We simply run the current version of Q across all possible a - if it's discrete, this is easy, just test every one.
If it's continuous, Q-learning is not a good fit.

To improve training stability, we train our network against a more stable version of the network,
that we only update to be in line with the current network every 10 or so iterations.

Another neat innovation DeepMind came up with is experience replay.
This simply takes the last N tuples (previous observation, action, observation, reward)
and trains the network on minibatches of this, instead of training it only on the most recent tuple.
This helps stabilize training as well.
