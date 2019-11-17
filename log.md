
Current output:
```
> p run_training.py --config experiments/dcgan.yml
Welcome to Dixit GAN!
Pytorch version: 1.3.1
Loading config: experiments/dcgan.yml
data {'path': 'examples/hymenoptera_data', 'workers': 2, 'image_size': 64, 'n_channels': 3}
model {'id': 'dcgan', 'z_dim': 100, 'generator': {'n_filters': 64}, 'discriminator': {'n_filters': 64}}
training {'gpus': 1, 'batch_size': 128, 'epochs': 5, 'lr': '2e-4', 'optimiser': {'id': 'adam', 'beta1': 0.5}}
Epoch 0
Done!
Epoch 1
Done!
...

```