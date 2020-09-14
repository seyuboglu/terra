# Terra
Terra is a package that enables reproducible data science workflows. 

## Using Terra with your project
Clone the terra repo:
```
git clone https://github.com/seyuboglu/terra.git
```

Create an environment and activate it:
```
conda create --name env_name python=3.8
conda activate env_name
```

Install terra in development mode:
```
pip install -e path/to/terra
```

Create a `terra_config.json` file for your project (you will need to ):
```
conda env config vars conda env config vars set TERRA_CONFIG_PATH="/afs/cs.stanford.edu/u/sabrieyuboglu/code/rotation/terra_config.json"
cond
```