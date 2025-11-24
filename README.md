## Supported python versions
python versions >=3.12,<3.14

## How to install

### Step 1 (recommended): Create new Conda enviroment or activate existing one 
#### If you use an existing environment make sure the python version is in the supported python versions.

##### Create a new conda environemnt and activate it
```shell
conda create --name <new_env> python=3.13
conda activate <new_env>
```

##### Or, activate existing environment
```shell
conda activate <existing_env>
```


### Step 2: Download package to local machine
Download the package (whl file):
```shell
curl -L https://github.com/Mila-MP/inPhormer/raw/refs/heads/main/dist/inphormer-0.1.6-py3-none-any.whl -O
```
You can also download the usage example notebook if you wish:
```shell
curl -L https://github.com/Mila-MP/inPhormer/raw/refs/heads/main/usage_example.ipynb -O
```

### Step 3: Install the package
Run
```shell
pip install inphormer-0.1.6-py3-none-any.whl
```
This might take a while (took ~10 minutes on a completely empty environemnt) depending on how many packages are already in your environment. 

### Step 4: Using the package
In python simply run
```python
import inphormer
```

### Example usage pyjupiter notebook
See the usage_example.ipynb file for examples on how to run the package. 
If you want to use the usage notebook, don't forget to install pyjupiter using 
```shell
pip install ipykernel
```


