---
layout: post
title: Jupyter-Utilities (for openai-gym)
author: Alessandro Santospirito
---

#### Debugging
```python
def save_params(func):
    def wrapper(*args, **kwargs):
        prefix = "_"
        globals().update({prefix + k: v for k, v in zip(func.__code__.co_varnames, args)})
        return func(*args, **kwargs)
    return wrapper

@save_params
def sum(a, b):
    return a + b

sum(1, 2)

print(f"a: {_a}, b: {_b}")
```

#### Render observation to Notebook
```python
import ipywidgets as widgets
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from IPython.display import display

image_widget = widgets.Image(format='jpeg')
display(image_widget)
```
```python
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

def render_game_to_cell(obs):
    screen_data = obs
    frame = cv2.cvtColor(screen_data, cv2.COLOR_RGB2BGR)
    _, jpeg = cv2.imencode('.jpeg', frame)
    image_widget.value = jpeg.tobytes()
```
```python
done = True

for step in range(1000):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    render_game_to_cell(state)

env.close()
```

#### Observation to mp4 (web-format)
```python
def convert_video_for_web(input_file, output_file):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx264',  # Video codec
        '-crf', '23',       # Constant Rate Factor (balance between quality and file size)
        '-preset', 'medium',  # Encoding speed preset
        '-c:a', 'aac',      # Audio codec
        '-b:a', '128k',     # Audio bitrate
        '-movflags', '+faststart',  # Optimize for web streaming
        '-y',               # Overwrite output file without asking
        output_file
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully converted {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print(f"ffmpeg stderr output:\n{e.stderr.decode()}")
```
```python
env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-v0"), SIMPLE_MOVEMENT)

frame_height, frame_width, _ = env.reset().shape
composite_height = frame_height
composite_width = frame_width
fps = 15
video_writer = cv2.VideoWriter('mario_recording.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (composite_width, composite_height))
episodes = fps * 15

done = True

for step in range(episodes):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    frame_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
    
    video_writer.write(frame_bgr)
    cv2.imshow('Composite Frame', frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_writer.release()
cv2.destroyAllWindows()
env.close()
```
```python
input_file = 'mario_recording.mp4'
output_file = 'mario_recording_web.mp4'

convert_video_for_web(input_file, output_file)
```
#### Four-Image Composite
```python
envs = [
    JoypadSpace(gym_super_mario_bros.make(f'SuperMarioBros-v{i}'), SIMPLE_MOVEMENT)
    for i in range(4)
]

def create_composite_frame(obs_list):
    height, width, _ = obs_list[0].shape
    
    composite = np.zeros((height*2, width*2, 3), dtype=np.uint8)
    
    composite[0:height, 0:width] = obs_list[0]
    composite[0:height, width:] = obs_list[1]
    composite[height:, 0:width] = obs_list[2]
    composite[height:, width:] = obs_list[3]
    
    return composite

frame_height, frame_width, _ = envs[0].reset().shape
composite_height = frame_height * 2
composite_width = frame_width * 2
fps = 15
video_writer = cv2.VideoWriter('mario_four_images_composite.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (composite_width, composite_height))

states = [env.reset() for env in envs]
dones = [False] * 4

episodes = fps * 15
for step in range(episodes):
    for i, env in enumerate(envs):
        if dones[i]:
            states[i] = env.reset()
        
        states[i], _, dones[i], _ = env.step(env.action_space.sample())
    
    composite = create_composite_frame(states)
    frame_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    
    video_writer.write(frame_bgr)
    cv2.imshow('Composite Frame', frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_writer.release()
cv2.destroyAllWindows()

for env in envs:
    env.close()
```
```python
input_file = 'mario_four_images_composite.mp4'
output_file = 'mario_four_images_composite_web.mp4'

convert_video_for_web(input_file, output_file)
```
