## Supported python versions
python versions >=3.12,<3.14

## How to install

### Step 1 Create Conda enviroment (optional)
##### if you use an existing environment make sure the python version is in the supported python versions)

Create a new conda environemnt
```bash
conda create --name inphormer python=3.13
conda activate inphormer
```

### Step 2: download whl file to local machine
download the whl file and the usage example notebook 
```shell
curl -L https://github.com/Mila-MP/inPhormer/raw/refs/heads/main/dist/inphormer-0.1.4-py3-none-any.whl -O
curl -L https://github.com/Mila-MP/inPhormer/raw/refs/heads/main/usage_example.ipynb -O
```


### Step 2: Install the package
Run ``pip install inphormer-0.1.4-py3-none-any.whl``

### Using the package
In python simply run
```python
import inphormer
```
See the usage_example.ipynb file for examples on how to run the package. 
