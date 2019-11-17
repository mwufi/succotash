from models import ModelBase, UpStack, DownStack


class DCGAN(ModelBase):
	name = 'dcgan'

	def __init__(self, config):
		super().__init__()
		g,d = config.generator, config.discriminator

		self.generator = UpStack(z_dim=config.z_dim, n_output_channels=3, n_base_filters=g.n_filters)
		self.discriminator = DownStack(n_data_channels=3, n_base_filters=d.n_filters)