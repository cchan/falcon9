# Landing a 2D Falcon 9 with RL
The Falcon 9 landing is [hilariously difficult for humans, even in 2D](https://www.youtube.com/watch?v=BPCUWebOick). This is because:

- It is a much taller and more top-heavy vehicle than the lunar lander, and its top RCS thrusters and gridfins have practically no control authority at these low speeds.
  - I've eliminated them entirely, so our simulated vehicle must depend entirely upon thrust vectoring and throttling of its main thruster.
- The Merlin engine can only throttle down to 39%, and can be turned off exactly once.
- There is a limited amount of fuel, which is spent in proportion to the throttle (roughly; it's probably less efficient at lower throttles?). (LunarLander-v2 assumes the throttle is either maximum or off, and can be switched arbitrarily many times with no fuel limit.)
- Vectoring of the main thruster can only happen at a fixed low rate. Throttling can probably happen relatively quickly, but not instantly either.
  - Based on watching a few videos of landings, thrust vectoring maximum angular rate is something on the order of 90 degrees per second.
  - Throttle rate is very difficult to tell from videos, so let's make it vary between 1.5 and 3.5 g's based on [this](https://en.wikipedia.org/wiki/Merlin_(rocket_engine_family)#Merlin_1D_Vacuum) and [this](https://www.quora.com/What-is-the-dry-mass-of-a-Falcon-9-FT), at ~0.5g per 0.1s based on [this](https://www.reddit.com/r/spacex/comments/5v4mxf/falcon_9_landing_strategy_analysed/ddzh2ka/)

Things I'm not taking into consideration (yet?):
- Fuel consumption decreases the mass of the vehicle
- Air resistance
- The actual Falcon 9 is 3D

## Action Space
- Thrust: vector at maximum rate left, vector maintain, vector at maximum rate right
- Throttle: throttle down at maximum rate, throttle maintain, throttle up at maximum rate, turn off throttle
Note that once the vehicle has turned off its throttle, it cannot be turned on again, and it is entirely at the mercy of gravity. This also happens if it runs out of fuel.

## Observation Space
- Current throttle position
- Current thrust vector position
- Current angle and angular rate
- Current position (relative to target position), velocity
- Fuel remaining

## End Conditions
We've seen Grasshopper blow up after turning upside down before, so I'll assume that once the rocket is horizontal it's a goner. Also, once it has hit the ground AND is turned off, it needs to be
1) vertical,
2) at a low velocity, and
3) at a very low angular rate.

## Reward
This is always the tricky part with reinforcement learning. We probably can't just do this:
- +100 reward for landing successfully and staying upright
    - A smaller reward for landing closer to the target
- -100 reward for not doing so

We need some more rewards to give the algorithm a few more hints as to what ideas are good without giving it a huge cliff to fall off of.
- Negative reward per frame for deviation from angle=0, x=0, y=0, v=0
- Negative reward per frame for spending fuel (proportional to throttle)
- Positive reward per frame for not dying (I think we need this to make sure it doesn't just get stuck trying to die immediately to minimize the above losses)

## Initial Conditions
This requires some calculation to make sure the task is actually physically possible. Taking the middle road between minimum of 1.5g and maximum of 3.5g throttle should probably do it - so we assume an acceleration of 2.5g to a stop at (0,0). If we want a 10 second long burn, we need to start it at $\frac{1}{2}at^2 \approx 1250\textrm{m}$ at a speed of $350\textrm{m/s}$ downward.

The position should be somewhat offset (100m?) from the landing pad, but it should be allowed to land anywhere for a start.

Also, start it with enough fuel for 10 seconds at max throttle (so something like 35 fuel points, subtracting throttle/FPS per frame.)
