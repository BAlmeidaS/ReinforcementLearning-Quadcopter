# Deep RL Quadcopter Controller

*Teach a Quadcopter How to Fly!*

This project is a solution to the final project proposed by Udacity in the Nanodegree - Advanced Machine Learning Engineer.

Udacity gives us a physic simulation written in python (physics_sim.py). In this file, we have a quadcopter (drone) with four helices, and a physics simulator which gives to the quadcopter an environment to fly.

My job is to propose and create a task for this quadcopter (task.py). I create a task that given some point that quadcopter is, it must go to another point. For doing this, I created a reward function, which could see in the task.py file. The reward function gives the agent a reward based on the distance from the point and its velocity. This problem was solved using the RL framework.

Besides that, I created an Agent using the DDPG strategy, and a bunch of utils functions and classes. All those files and content could be founded in `agents/DDPG` folder.

I have also created a grid search file useful to find the best hyperparameters using [ray](https://github.com/ray-project/ray) to create multiprocess branchs to find the best parameters. In the folders `weights*` we could see some graphs of rewards and which parameters used to draw these graphs, besides the weights of the networks trained.

Last but not least, we have a Jupyter Notebook which gives a full context and the full exploration of the problem called `QuadcopterProject.ipynb`. If you come here from somewhere, I hope you enjoy to meet this challenge with my solution =].

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.

```
git clone https://github.com/udacity/RL-Quadcopter-2.git
cd RL-Quadcopter-2
```

2. Create and activate a new environment.

```
conda create -n quadcop python=3.6 matplotlib numpy pandas
source activate quadcop
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `quadcop` environment. 
```
python -m ipykernel install --user --name quadcop --display-name "quadcop"
```

4. Open the notebook.
```
jupyter notebook Quadcopter_Project.ipynb
```

5. Before running code, change the kernel to match the `quadcop` environment by using the drop-down menu (**Kernel > Change kernel > quadcop**). Then, follow the instructions in the notebook.

6. You will likely need to install more pip packages to complete this project.  Please curate the list of packages needed to run your project in the `requirements.txt` file in the repository.

### Useful commands
run grid search in the terminal:
`$ python3 grid_search.py --num_of_cores 8`
