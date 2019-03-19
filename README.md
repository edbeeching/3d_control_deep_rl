# 3D Control and Reasoning without a Supercomputer
A collection of scenarios and efficient benchmarks for the ViZDoom RL environment.

This repository includes:
* Source code for generation of custom scenarios for the ViZDoom simulator
* Source code for training new agents with the GPU-batched A2C algorithm
* Detailed instructions of how to evaluate pretrained agents and train new ones
* Example videos of rollouts of the agent.


# Contents
* [Installation](https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks/blob/master/README.md#installation-1)
* [Custom scenario generation](https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks#custom-scenario-generation-1)
* [Evaluating and training agents](https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks#evaluating-pretrained-agents-and-training-new-ones-on-the-scenarios)
* [FAQ](https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks#faq)
* [Citation TODO](https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks#citation-1)
* [Acknowledgements TODO](https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks/blob/master/README.md#acknowledgements)

## Installation
### Requirements

* Ubuntu 16.04+ (there is no reason this will not work on Mac or Windows but I have not tested)
* python 3.5+
* PyTorch 0.4.0+ 
* ViZDoom 1.1.4 (if evaluating a pretrained model, otherwise the latest vision should be fine)

### Instructions
1. [ViZDoom](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_deps) has many dependencies which are described on their site, make sure to install the [ZDoom](https://zdoom.org/wiki/Compile_ZDoom_on_Linux)  dependencies.
2. Clone this repo
3. Assuming you are using a venv, activate it and install packages listed in requirements.txt
4. Test the installation with the following command, this should train an agent 100,000 frames in the basic health gathering scenario:
```
    python  train_a2c.py  --num_frames 100000
```
Note if you want to train this agent to convergence it takes between 5-10M frames.

## Custom scenario generation
As detailed in the paper, there are a number of scenarios.
We include a script **generate_scenarios.sh** in the repo that will generate the following scenarios:
 
* Labyrinth: Sizes 5, 7, 9, 11, 13
* Find return: Size 5, 7, 9, 11, 13
* K-Item : 2, 4, 6, 8 items
* Two color correlation: 10%, 30%, 50% and 70% walls retained.

This takes around 10 minutes so grab a coffee. If you wish to only generate for one scenario, take a look at the script it should be clear what you need to change.

## Evaluating pretrained agents and training new ones on the Scenarios

We include pretrained models in the repo that you can test out, or you can train your own agents from scratch.
The evaluation code will output example rollouts for all 64 test scenarios.

### Labyrinth

<p float="left">
  <img src="https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks/blob/master/videos/labyrinth_example.gif" height="200" />
  <img src="https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks/blob/master/results/labyrinth_res.png" height="200" /> 
</p>

Evaluation:
```
SIZE=9
python create_rollout_videos.py --recurrent_policy --num_stack 1 --limit_actions \
       --scenario_dir  scenarios/custom_scenarios/labyrinth/$SIZE/test/ \
       --scenario custom_scenario{:003}.cfg  --model_checkpoint \
       saved_models/labyrinth_$SIZE\_checkpoint_0198658048.pth.tar \
       --multimaze --num_mazes_test 64
```

Training:
```
SIZE=9
python  train_a2c.py --scenario custom_scenario{:003}.cfg \ 
        --recurrent_policy --num_stack 1 --limit_actions \
        --scenario_dir scenarios/custom_scenarios/labyrinth/$SIZE/train/ \
        --test_scenario_dir scenarios/custom_scenarios/labyrinth/$SIZE/test/ \
        --multimaze --num_mazes_train 256 --num_mazes_test 64 --fixed_scenario
```

### Find and return
<p float="left">
  <img src="https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks/blob/master/videos/find_return_example.gif" height="200" />
  <img src="https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks/blob/master/results/find_return_results.png" height="200" /> 
</p>

Evaluation:
```
SIZE=9
python create_rollout_videos.py --recurrent_policy --num_stack 1 --limit_actions \
       --scenario_dir  scenarios/custom_scenarios/find_return/$SIZE/test/ \
       --scenario custom_scenario{:003}.cfg  --model_checkpoint \
       saved_models/find_return_$SIZE\_checkpoint_0198658048.pth.tar \
       --multimaze --num_mazes_test 64
```
Training:
```
SIZE=9
python  train_a2c.py --scenario custom_scenario{:003}.cfg \
        --recurrent_policy --num_stack 1 --limit_actions \
        --scenario_dir scenarios/custom_scenarios/find_return/$SIZE/train/ \
        --test_scenario_dir scenarios/custom_scenarios/find_return/$SIZE/test/ \
        --multimaze --num_mazes_train 256 --num_mazes_test 64 --fixed_scenario
```

### K-item
<p float="left">
  <img src="https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks/blob/master/videos/4_item_example.gif" height="200" />
  <img src="https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks/blob/master/results/kitem_res.png" height="200" /> 
</p>

Evaluation:
```
NUM_ITEMS=4
python create_rollout_videos.py --recurrent_policy --num_stack 1 --limit_actions \
       --scenario_dir  scenarios/custom_scenarios/kitem/$NUM_ITEM/test/ \
       --scenario custom_scenario{:003}.cfg  --model_checkpoint \
       saved_models/$NUM_ITEMS\item_checkpoint_0198658048.pth.tar \
       --multimaze --num_mazes_test 64
```
Training:
```
NUM_ITEMS=4
python  train_a2c.py --scenario custom_scenario{:003}.cfg \
        --recurrent_policy --num_stack 1 --limit_actions \
        --scenario_dir scenarios/custom_scenarios/kitem/$NUM_ITEMS/train/ \
        --test_scenario_dir scenarios/custom_scenarios/kitem/$NUM_ITEMS/test/ \
        --multimaze --num_mazes_train 256 --num_mazes_test 64 --fixed_scenario
```

### Two color correlation
<p float="left">
  <img src="https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks/blob/master/videos/two_color_example.gif" height="200" />
  <img src="https://github.com/edbeeching/3D_Control_RL_Scenario_Benchmarks/blob/master/results/two_color_correlation_res.png" height="200" /> 
</p>

Evaluation:
```
DIFFICULTY=3
python create_rollout_videos.py --recurrent_policy --num_stack 1 --limit_actions \
       --scenario_dir  scenarios/custom_scenarios/two_color/$DIFFICULTY/$DIFFICULTY/test/ \
       --scenario custom_scenario{:003}.cfg  --model_checkpoint \
       saved_models/two_col_p$DIFFICULTY\_checkpoint_0198658048.pth.tar \
       --multimaze --num_mazes_test 64
```
Training:

```
DIFFICULTY=3
python  train_a2c.py --scenario custom_scenario{:003}.cfg \
        --recurrent_policy --num_stack 1 --limit_actions \
        --scenario_dir scenarios/custom_scenarios/two_color/$DIFFICULTY/train/ \
        --test_scenario_dir scenarios/custom_scenarios/two_color/$DIFFICULTY/test/ \
        --multimaze --num_mazes_train 256 --num_mazes_test 64 --fixed_scenario
```


## FAQ
### Why is my FPS 4x lower than in your paper?
In the paper we report a frames per second in terms of envionrment interactions, the agents are trained with a frame skip of 4, which means for each observation the same action is repeated 4 times.

### I have limited memory, is there anything I can do?
Yes, we have made a tradeoff between increased memory usage in order to increase performance, you can reduce the memory footprint by excluding **--fixed scenario** from the command line arguments. You will see a 10% drop in efficiency.

### 


## Citation
Please cite the following:
TODO upon acceptance

## Acknowledgements
TODO upon acceptance




