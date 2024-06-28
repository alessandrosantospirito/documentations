---
layout: post
title: Supermario-RL - Frame-Stacking, PPO
author: Alessandro Santospirito
---

#### Supermario Enviroment
```python
# Setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0') # options: v0, v1, v2, v3
env = JoypadSpace(env, SIMPLE_MOVEMENT)
```

<div class="video-container">
    <video src="{{ site.url }}/images/supermario-rl/mario_four_images_composite_web.mp4" alt="supermario version" autoplay loop></video>
</div>

#### Frame-Stacking
We will stack multiple (gray-scaled) frames into one observation such that `env.reset()` or `obs` has shape `(1, 240, 256, 4)`.
The interpretation is as follows:
- `1`: one vector we train on
- `240, 256`: the gray-level of each pixel
- `4`: four frames in one observation

```python
# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')
```

<div class="video-container">
    <video src="{{ site.url }}/images/supermario-rl/mario_four_gray_images_composite_web.mp4" alt="supermario version" autoplay loop></video>
</div>

#### Train the RL-Model
```python
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512) 
```

#### Test the Model
```python
model = PPO.load('./train/best_model_1000000')

state = env.reset()
done = False

while not done: 
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()

env.close()
```
