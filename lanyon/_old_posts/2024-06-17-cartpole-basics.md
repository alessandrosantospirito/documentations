---
layout: post
title: Cartpole - Basics
---

#### Setup
```python
import time
import gym

env_name = 'CartPole-v0'
env = gym.make(env_name)
```
#### Main-Loop
```python
done = True

for step in range(1000):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    time.sleep(0.1)

env.close()
```

#### Render
<!-- ![Cart Pole - Random Actions as input](<./pytorch-rl/cartpole/md/assets/cart-pole-basic.gif>) -->
