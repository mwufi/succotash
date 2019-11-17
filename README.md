# WOP

Current output:
```
> p run_training.py --config experiments/dcgan.yml
Welcome to Dixit GAN!
Pytorch version: 1.3.1
Loading config: experiments/dcgan.yml
data {'path': 'examples/hymenoptera_data', 'workers': 2, 'image_size': 64, 'n_channels': 3, 'batch_size': 128}
model {'id': 'dcgan', 'z_dim': 100, 'generator': {'n_filters': 64}, 'discriminator': {'n_filters': 64}}
DCGAN(
  (generator): UpStack(
    (main): Sequential(
      (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
....
    )
  )
  (discriminator): DownStack(
    (main): Sequential(
      (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): LeakyReLU(negative_slope=0.2, inplace=True)
      (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
 ...
      (12): Sigmoid()
    )
  )
)
Going to start training soon!
...

```

It also shows you a window:
<img src="https://raw.githubusercontent.com/mwufi/succotash/master/docs/batch.png" width="75%">
