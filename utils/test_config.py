import unittest
from pathlib import Path

from utils import Config


class TestConfig(unittest.TestCase):
	def test_loading_same_directory(self):
		c = Config(Path.cwd() / 'utils' / 'config.yml')
		self.assertEqual(c.mysql.host, 'localhost')
		self.assertEqual(c.other.preprocessing_queue[0], 'preprocessing.scale_and_center')

	def test_loading_experiment(self):
		c = Config(Path.cwd() / 'experiments' / 'sample.yml')
		self.assertEqual(c.model.bias, True)
		self.assertEqual(c.training.location, 'aws.google.com')
		self.assertEqual(c.training.alert.every, '20 minutes')
