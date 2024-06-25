---
layout: post
title: Cartpole - Arctor Critic Methode
---

#### Math
$$
x^2
$$

##### Adam
Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.

- [keras - Adam](https://keras.io/api/optimizers/adam/)

#### Setup
```python
rl_model = RL()
lr = 1e-3
optimizer = optim.Adam(rl_model.parameters(), lr=lr)

max_steps = 100000
rollouts = 0
step = 0
score_logger = []
```
##### Main-Loop
```python
while step < max_steps:
    observation =  torch.FloatTensor(env.reset()).unsqueeze(0)
    done = False
    rewards = []
    values = []
    log_probs = []
    
    while not done:
        dist, value = rl_model(observation)
        action = dist.sample()
        log_prob = dist.log_prob(action.unsqueeze(0))
        
        observation, reward, done, info = env.step(action.cpu().item())
        env.render()
        time.sleep(0.05)
        
        observation = torch.FloatTensor(observation).unsqueeze(0)
        reward = torch.FloatTensor([reward]).unsqueeze(0)

        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        step +=1
    
    returns = calc_returns(rewards)
    
    returns = torch.cat(returns, 1)
    log_probs = torch.cat(log_probs, 1)
    values = torch.cat(values, 1)
    advantage = (returns - values).detach()
    
    action_loss = - (log_probs * advantage).mean()
    critic_loss = (returns - values).pow(2).mean()
    agent_loss = action_loss + critic_loss
    
    optimizer.zero_grad()
    agent_loss.backward()
    optimizer.step()
    rollouts += 1
    
    if rollouts % 10 == 0:
        new_lr = ((max_steps - step)/max_steps) * lr
        optimizer.param_groups[0]["lr"] = new_lr
        
        score_logger.append(np.mean([test_agent() for _ in range(10)]))
        clear_output(True)
        plt.plot(score_logger)
        plt.show()
    
env.close()
```
#### Render


