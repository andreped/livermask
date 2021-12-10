#coding:utf-8
import shutil
import sys, os, time
import yaml
from utils import yaml_utils


#sys.path.append(os.path.dirname(__file__))

# Copy from tgans repo.
class Config(object):
    '''
    'https://github.com/pfnet-research/sngan_projection/blob/master/source/yaml_utils.py'
    '''
    def __init__(self, config_dict):
        self.config = config_dict

    def __getattr__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise AttributeError(key)

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)
