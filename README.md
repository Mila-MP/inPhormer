# inPhormer: a Python Package to Predict Phage Protein Annotation Information Content
**inPhormer** is a Python package designed to assess and categorize the information content of protein annotations for bacteriophages (phages). High-quality, informative annotations are critical for biological discovery, but many existing phage protein entries suffer from poor or non-specific descriptions.

This tool provides a rapid, reliable, and scalable way to classify annotations into three distinct categories:

- **Uninformative** (0): Annotations that provide no functional insight (e.g. "Hypothetical protein", "Uncharacterized protein").
- **Low Informative** (1): Annotations that offer some general context but lack specificity (e.g. "3D (Asp-Asp-Asp) domain-containing protein", "DUF3365 domain-containing protein").
- **Proper** (2): Annotations that contain sufficient, specific functional detail (e.g. "Nucleosome remodeling complex atpase subunit", "Head-tail connector protein").

The package gives the option to choose between three different prediction methods:
1. **Term Frequency-Inverse Document Frequency** (TF-IDF) trained on phage protein annotation data
2. [BioBERT encoder](https://huggingface.co/pritamdeka/S-BioBert-snli-multinli-stsb) with a classification head trained on phage protein annotation data
3. [[PubMedBERT encoder](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) + classification layer] **fine-tuned** on phage protein annotation data

The three methods in inPhormer offer a trade-off between speed and predictive accuracy. The TF-IDF method is the fastest and most computationally lightweight, making it ideal for extremely rapid, large-scale preliminary filtering, though it offers the lowest overall accuracy. Conversely, the fine-tuned approach is the most accurate method, but it requires the longest computation time. The second method (^re-trained encoder with classification head strikes a good balance between the two.

# Installation
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


### Step 3: Install the package
Run
```shell
pip install inphormer-0.1.6-py3-none-any.whl
```
This might take a while (~10 minutes) depending on how many packages are already in your environment and/or cached packages. 

### Step 4: Using the package
In python simply run
```python
import inphormer
```

### Example usage Jupyter notebook
You can download the usage example notebook if you wish:
```shell
curl -L https://github.com/Mila-MP/inPhormer/raw/refs/heads/main/usage_example.ipynb -O
```
Don't forget to install ipykernel in your environment to be able to run the notebook:
```shell
pip install ipykernel
```


