import yaml
import argparse
import warnings
from huggingface_hub import HfApi
# import logging
# from logger_config import init_logger

def parse_yaml_config(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{yaml_file}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    
    return config or {}

def get_base_model_from_hf(model_path):
    try:
        api = HfApi()
        info = api.model_info(model_path)
        
        if info.card_data and info.card_data.get('base_model'):
            base_spec = info.card_data.get('base_model')
            if isinstance(base_spec, list) and base_spec:
                return base_spec[0]
            elif isinstance(base_spec, str):
                return base_spec
    except Exception as e:
        warnings.warn(f"Could not get base model for {model_path}: {e}")
    
    return None

def determine_tokenizer_for_model(model_config):
    model_path = model_config['path']
    if 'tokenizer_path' in model_config and model_config['tokenizer_path']:
        return model_config['tokenizer_path']
    
    # TODO: Buscar pelo base_model não é interessante pq o modelo pode ser base sem chat_template
    # base_model = get_base_model_from_hf(model_path)
    # if base_model and base_model != model_path:
    #     return base_model
    
    return model_path

def get_config_with_defaults(yaml_config):    
    defaults = {
        'num_shots': 5,
        'num_experiments': 3,
        'multi_gpu': {
            'enabled': False,
            'num_gpus': 1
        },
        'flash_attention': False,
        'update_leaderboard': False,
        'run_local': False,
        'benchmark_names': [
            "assin2rte",
            "assin2sts", 
            "bluex",
            "enem",
            "hatebr",
            "portuguese_hate_speech",
            "faquad",
            "tweetsentbr",
            "oab"
        ]
    }
    
    required_fields = ['run_id', 'model_paths']
    
    config = defaults.copy()
    config.update(yaml_config)

    valid_backends = ['hf', 'vllm']
    if config['backend'] not in valid_backends:
        raise ValueError(f"'backend' must be one of {valid_backends}, got: {config['backend']}")
    
    if not isinstance(config.get('multi_gpu'), dict):
        raise ValueError("'multi_gpu' must be a dictionary with 'enabled' and 'num_gpus' fields")
    
    config['multi_gpu'].setdefault('enabled', False)
    config['multi_gpu'].setdefault('num_gpus', 1)
    
    missing_fields = [field for field in required_fields if field not in config or config[field] is None]
    if missing_fields:
        raise ValueError(f"Missing required fields in configuration: {missing_fields}")
    
    if not isinstance(config['model_paths'], list) or len(config['model_paths']) == 0:
        raise ValueError("'model_paths' must be a non-empty list")
    
    processed_model_paths = []
    for model in config['model_paths']:
        if isinstance(model, str):
            model_config = {'path': model, 'custom': False}
        elif isinstance(model, dict):
            if 'path' not in model:
                raise ValueError("Model dict must contain 'path' field")
            model_config = model.copy()
            model_config.setdefault('custom', False)
        else:
            raise ValueError("Model paths must be strings or dicts with 'path' field")
        
        model_config['tokenizer_path'] = determine_tokenizer_for_model(model_config)
        processed_model_paths.append(model_config)
    
    config['model_paths'] = processed_model_paths
    
    if not isinstance(config['benchmark_names'], list) or len(config['benchmark_names']) == 0:
        raise ValueError("'benchmark_names' must be a non-empty list")
    
    return config

def generate_bash_variables(config):    
    bash_vars = []
    
    bash_vars.append(f'NUM_SHOTS={config["num_shots"]}')
    bash_vars.append(f'NUM_EXPERIMENTS={config["num_experiments"]}')
    bash_vars.append(f'RUN_ID="{config["run_id"]}"')
    bash_vars.append(f'BACKEND="{config["backend"]}"')  # Add backend variable
    bash_vars.append(f'MULTI_GPU_ENABLED={str(config["multi_gpu"]["enabled"]).lower()}')
    bash_vars.append(f'MULTI_GPU_NUM_GPUS={config["multi_gpu"]["num_gpus"]}')
    bash_vars.append(f'FLASH_ATTENTION_ENABLED={str(config["flash_attention"]).lower()}')  # Add flash attention variable
    bash_vars.append(f'UPDATE_LEADERBOARD={str(config["update_leaderboard"]).lower()}')
    bash_vars.append(f'RUN_LOCAL={str(config["run_local"]).lower()}')
    bash_vars.append('MODEL_PATHS=(')
    for model in config['model_paths']:
        bash_vars.append(f'  "{model["path"]}"')
    bash_vars.append(')')
    
    bash_vars.append('MODEL_CUSTOM_FLAGS=(')
    for model in config['model_paths']:
        bash_vars.append(f'  {str(model["custom"]).lower()}')
    bash_vars.append(')')
    
    bash_vars.append('MODEL_TOKENIZERS=(')
    for model in config['model_paths']:
        bash_vars.append(f'  "{model["tokenizer_path"]}"')
    bash_vars.append(')')
    
    bash_vars.append('')
    bash_vars.append('BENCHMARK_NAMES=(')
    for benchmark in config['benchmark_names']:
        bash_vars.append(f'  "{benchmark}"')
    bash_vars.append(')')
    
    return '\n'.join(bash_vars)

def main():
    parser = argparse.ArgumentParser(description='Parse YAML configuration for run.sh script')
    parser.add_argument('config_file')
    args = parser.parse_args()
    # init_logger()
    
    yaml_config = parse_yaml_config(args.config_file)
    config = get_config_with_defaults(yaml_config)
    
    bash_output = generate_bash_variables(config)
    print(bash_output)

if __name__ == '__main__':
    main()