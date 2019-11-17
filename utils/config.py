from pathlib import Path

import yaml


class Config:
	def __init__(self, filename: str):
		self.dict = dict()

		if not filename:
			return

		with open(filename, 'r') as ymlfile:
			cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
			print('Loaded...', cfg)
			for k, v in cfg.items():
				self.add_item(k, Config.from_dict(v))

	def add_item(self, k, v):
		setattr(self, k, v)
		self.dict[k] = v

	@staticmethod
	def from_dict(config_object):
		literals = (float, int, str)
		if isinstance(config_object, literals):
			return config_object
		elif isinstance(config_object, list):
			return config_object
		elif isinstance(config_object, dict):
			c = Config(None)
			for k, v in config_object.items():
				c.add_item(k, Config.from_dict(v))
			return c

	def __repr__(self):
		return self.dict.__repr__()


def test_config():
	print('Current working dir:', Path.cwd())
	c = Config(Path.cwd() / 'utils' / 'config.yml')
	print('c.mysql', c.mysql)
	print('c.other.preprocessing_queue', c.other.preprocessing_queue)
