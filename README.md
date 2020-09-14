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

Create a `terra_config.json` file for your project:
```
# terra_config.json
{
    "storage_dir": "path/to/storage/dir",
    "slack_web_client_id": "xoxb...",
    "notify": true
}

```

Then set the `TERRA_CONFIG_PATH` variable in your environment to point to your new `terra_config.json` (you'll need to reactivate the environment): 
```
conda env config vars conda env config vars set TERRA_CONFIG_PATH="path/to/terra_config.json"
conda activate env_name
```
