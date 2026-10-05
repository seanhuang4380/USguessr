# USguessr

ML classification model for the US

https://huggingface.co/seanhua4/USguessr/tree/main
<br><br>

Based on Resnet architecture. 
<br><br>
Loss: 2.3633

Accuracy: 0.40
<br><br>

Hyperparameters used during training:
> train_batch_size: 32 

> eval_batch_size: 32 

> optimizer: Adam with (default) betas=(0.9,0.999) and epsilon=1e-08

> lr_scheduler_type (default): linear

> Layer 4 learning rate: e-5

> FC head learning rate: e-4

> num_epochs: 7 