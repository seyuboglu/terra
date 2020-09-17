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

Create a `terra_config.json` file for your project.

Set an environment variable in your conda environment pointing to your `terra_config.json` (you will need to reactivate the environment after setting the variable):
```
conda env config vars set TERRA_CONFIG_PATH="/path/to/terra_config.json"
conda deactivate
conda activate env_name
```