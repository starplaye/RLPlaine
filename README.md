# RLPlane

## Install 

In the main folder rlplane :
```git clone git@git.unistra.fr:a.gautheron/rlplane.git && cd rlplane && python3 -m venv venv && source ./venv/bin/activate && pip install -r requirements.txt```

## Launch

You can launch all experimentations with the ```*_train.py``` files.

The arguments are :
- ```-m``` for the mode ( vRL / HRL)
- ```--max_iterations``` for the number of iterations to train your policy
- ```-w``` for the path the the worker file if you use a pre-train worker in HRL mode
- ```-B``` the number of environnments parallelized
- ```-e``` to name your experimentation in a specific way

Run file command : 
```python3 environnements/maze/run.py```



