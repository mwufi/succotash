
Current output:
```b
/Users/zentang/projects/untitled [git::master *] [zentang@Not-Just-Osiris] [15:22]
> p run_training.py
Welcome to Dixit GAN!
Pytorch version: 1.3.1

-----------------(init)-----------------
MLP
2410 trainable parameters
-------------(end of init)--------------


-----------------(init)-----------------
bottleneck
190 trainable parameters
-------------(end of init)--------------

:202 - bottleneck - input size: torch.Size([13, 3, 40, 2])
:204 - bottleneck - output size: torch.Size([13, 5, 40, 2])
:204 - MLP - input size: torch.Size([13, 240])
:207 - MLP - output size: torch.Size([13, 10])

```