import json, os, yaml


class ExtUtil:
    def __init__(self):
        super().__init__()

    @staticmethod
    def load_json_conf(conf_path):
        # with open('/data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/qwen_ext/config/qwen_special_tokens.json',
        #           'r') as file:
        with open(conf_path, 'r') as f:
            json_conf = json.load(f)
        return json_conf

    @staticmethod
    def load_yaml_conf():
        conf_path = r'config/conf_qwen_ext.yaml'
        with open(os.path.dirname(__file__) + conf_path, 'r') as s1_yaml:
            yaml_conf = yaml.safe_load(s1_yaml)
        return yaml_conf
