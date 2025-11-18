### Step 1: Install poetry  
Install poetry using ``pipx install poetry``  
Check version: ``poetry --version``  
More details [here](https://python-poetry.org/docs/).

### Step 2: Clone repo to local machine
Clone this repo on your local machine  

### Step 3: Install the poetry environment
Install the poetry environment to get all the dependencies using ``poetry install``  

### Step 4: Activate poetry environment
Activate poetry environment using ``poetry env activate``  
This will print something like ``"C:\Users\...\Scripts\activate.bat"``.
Copy this to your command line and enter. This should activate the poetry environment. 
If instead using bash, ``poetry env activate``, 
copy the output and run ``source 'C:\Users\...\Scripts\activate'``.

### Step 5: Install pytorch 
Install pytorch [here](https://pytorch.org/get-started/locally/).


## Or, install using wheel 
### Step 1: Clone repo to local machine
clone this repo on your local machine

### step 2: install the package (ideally inside a conda env)
pip install inPhormer\dist\inphormer-0.1.2-py3-none-any.whl

### step 3 Install pytorch
Install pytorch [here](https://pytorch.org/get-started/locally/).

### use the package
in python simply 
```python
import inphormer
```
