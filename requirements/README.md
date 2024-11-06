


```shell
conda create -y --name deepjoint_algo python=3.10
conda activate deepjoint_algo

pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 -r requirements/requirements.txt -r requirements/requirements-dev.txt
```