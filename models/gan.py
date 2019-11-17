from models import ModelBase


class DCGAN(ModelBase):
	def _init__(self):
		super().__init__()
		self._post_init()
