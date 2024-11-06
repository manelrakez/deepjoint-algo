# The DeepJoint algorithm
The codes for our proposed model, the "DeepJoint algorithm", which aims to quantitavely estimate mammographic density from the complete exam at each screening visit and evaluate its association with breast cancer risk over time, are provided here.
Refer to our [paper](https://arxiv.org/abs/2403.13488v1) for more details on the model.

Pytorch, R and all the other depencies are installed within the docker images `deepjoint_torch` and `deepjoint_r`. 
You only need to have Docker Engine and Python 3 installed locally to build and run the docker images.

### 1/ Requirements 
* Linux-based OS 
* Install [Docker Engine](https://docs.docker.com/get-started/get-docker/) and [Python 3](https://www.python.org/downloads/)

### 2/ Create the docker image:

- **deepjoint_torch** docker image
```
cd deepjoint-algo/ # root of the project
docker build --pull --no-cache -f docker/deepjoint_torch/Dockerfile -t deepjoint_torch:latest .
```
- **deepjoint_r** docker image
```
cd deepjoint-algo/ # root of the project
docker build --pull --no-cache -f docker/deepjoint_r/Dockerfile -t deepjoint_r:latest .
```

### 3/ Launch Bash prompt or Jupyter Notebook in a container

The file `~/deepjoint-algo/docker/run_in_docker.sh` helps you to run the container with good 
permissions and mounts the folders you need into the container.

From the docker images built above you can run a Bash prompt to run training & inference scripts (see below) or 
start a Jupyter notebook for interactive analysis.

- the **deep-learning component** of the DeepJoint algorithm

```
bash ~/deepjoint-algo/docker/run_in_docker.sh -i deepjoint_torch:latest  bash
```

```
bash ~/deepjoint-algo/docker/run_in_docker.sh -i deepjoint_torch:latest jupyter <PORT>
```

- the **joint model** part in the DeepJoint algorithm
```
bash ~/deepjoint-algo/docker/run_in_docker.sh -i deepjoint_r:latest bash
```

```
bash ~/deepjoint-algo/docker/run_in_docker.sh -i deepjoint_r:latest jupyter <PORT>
```

### 4/ Run the DeepJoint Components

The complete pipeline of the DeepJoint algorithm is run as follows:
* Train the Deep-learning model
* Make inferences
  * Optional run evaluation
* Fit the joint model
* Run individual predictions

You can either re-train the model and carry on with an inference step,
or directly start from the inference step and move forward to the joint model part.

NB: All the below `.sh` scripts in `scripts/` folder may be updated to fit to you configuration 
(GPUs, paths to your datasets and annotations).

Start from a running container with docker image `deepjoint_torch:latest` for steps 1 & 2.
Use `deepjoint_r:latest` image for steps 3 & 4 (see section 3/ to start the containers)


**Requirements**: All the next steps require HDF5 files. You have to launch the DICOM->HDF5 extraction before. 

```shell
# To be executed in `deepjoint_torch:latest` container
bash ~/deepjoint-algo/scripts/deepjoint_torch/launch_dcm_to_h5.sh
```


#### 1. Train the deep-learning model

Train the deep-learning model using the pre-trained version.

```
bash ~/deepjoint-algo/scripts/deepjoint_torch/launch_train.sh
```

#### 2. Make inferences

Make inferences for image-level mammographic density metrics and compute them to the visit-level
```
bash ~/deepjoint-algo/scripts/deepjoint_torch/launch_infer.sh
```

Optionally, you can evaluate the deep-learning model on data with annotations
(same format as during the training) with :

```
bash ~/deepjoint-algo/scripts/deepjoint_torch/launch_eval.sh
```

#### 3. Fit the joint model

```
bash ~/deepjoint-algo/scripts/deepjoint_r/launch_ModelFit.sh
```

#### 4. Run individual predictions (with cross-validation)

```
bash ~/deepjoint-algo/scripts/deepjoint_r/launch_DynPred_withCrossVal.sh
```

