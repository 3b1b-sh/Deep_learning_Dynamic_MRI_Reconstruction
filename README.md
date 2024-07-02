# Deep learning Dynamic MRI Reconstruction

### 1. Accelerated Cine Imaging with Aliasing

The undersampling mask is generated separately for each dynamic frame. We can run following code to get the mask of 20 frames dynamic image.

```python
from CS_mask import cartesian_mask
mask = cartesian_mask(shape=(1, 20, 192, 192), acc=6, sample_n=10, centred=True)
imsshow(mask[0, :, :, :], num_col=5, cmap='gray')
```

The output is at Graph 1.1. We can clearly see that the undersampling mask for different dynamic frames are different.

Then, we can obtain the aliased images as a result of undersampling with the generated patterns, according to the formula: $b=F^{-1}UFm$, where $m$ is the fully sampled image, $U$ is the corresponding undersampling pattern, $F$ is the Fourier transform and $b$ is the aliased image.

The aliased images(Graph 1.2) and fully sampled image(Graph 1.3) are as following.

| **Graph 1.1 undersampling mask for different dynamic frames**             |
| ------------------------------------------------------------------------------- |
| ![image](https://cdn.luogu.com.cn/upload/image_hosting/d9163zht.png)              |
| **Graph 1.2 Aliased images**                                              |
| ![testundersampling7](https://cdn.luogu.com.cn/upload/image_hosting/j7rsdvge.png) |
| **Graph 1.3 fully sampled image**                                         |
| ![testfullsampling7](https://cdn.luogu.com.cn/upload/image_hosting/92uhwwav.png)  |

### 2. Basic Network

We split the cine dataset into training, validation and testing with a ratio of 4:1:2, which is separately 114, 29 and 57. The code is as following.

```python
train_size = 114
val_size = 29
test_size = 57
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    dataloader_test =  DataLoader(test_set, batch_size=batch_size, shuffle=True)
```

To explore the temporal correlation, we choose to stack the dynamic images along the channel dimension so that the input dimention is 20, which is the frames of the dynamic image. By the way, we have tried 3D CNN. But it's so memory comsuming that we finally give up this idea.

However, Stacking also brings some problems. The image is pseudo complex, which means each frame has two dimention. Thus, it's strange to stack them directly to utilize temporal correlation. After several trials, we find processing real part and Imaginary part works best.

We choose U-Net as backbone and the model is as following:

![无标题的笔记本 3](https://cdn.luogu.com.cn/upload/image_hosting/g8b899wj.png)

Noticed that the two U-Net don't share parameters bucause it will decrease the performance of the model. The reason for that may be the Peculiarities of data.

However, we find that the performance of our model is not good enough. Also, the train loss and eval loss are very close, which show that our model maybe can be more complex to improve the performance. Instead of only increase the parameters of U-Net, we add another ResNet network after getting Ouptut of U-Net. The final model is as following:![无标题的笔记本 4](https://cdn.luogu.com.cn/upload/image_hosting/vqnxwo79.png)

The best performance of our model is as following:

| Loss              | Avg PSNR         | Std PSNR | Avg SSIM        | Std SSIM |
| ----------------- | ---------------- | -------- | --------------- | -------- |
| **0.00166** | **28.214** | 2.057    | **0.815** | 0.035    |

A good reconstructed dynamic image is as following:

| **Before reconstruction**                                                  |
| -------------------------------------------------------------------------------- |
| ![testundersampling7](https://cdn.luogu.com.cn/upload/image_hosting/j7rsdvge.png)  |
| **After reconstruction**                                                   |
| ![testreconstruction7](https://cdn.luogu.com.cn/upload/image_hosting/3rrds8rh.png) |

We can see that the reconstructed image is veery clear with much details, though the upper right corner of the picture is slightly blurry.

### 3. Optimization Techniques

Dropout is a widely used regularization technique in neural networks to prevent overfitting. Basically, adding dropout will improves the network's generalization ability, making it perform better on unseen data. We also tried to add dropout to our model. However, it doesn't work well. It ruined the performance of our model. We guess it‘s because that the images in the dataset have similar distribution. Thus, model without strong regularization can also works well and dropout even disturb the training process which makes the network hard to train.

At the same time, we tried data agumentation. However, it doesn't works well, too. The code is as following:

```python
def apply_transforms(x):
    batch_size, num_samples, channels, height, width = x.size()
    x_flat = x.view(-1, channels, height, width)
    transformed_samples = []
    for sample in x_flat:
        sample_pil = transforms.ToPILImage()(sample)
        sample_transformed = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ToTensor()
        ])(sample_pil)
        transformed_samples.append(sample_transformed)
    transformed_x = torch.stack(transformed_samples).view(batch_size, num_samples, channels, height, width)
    return transformed_x
```

We tried single and many combinations of these transformations but there is no even one tranformation finally improves the performance of our model. We suspect that the dataset is small and such transformations make the model harder to capture the features of images, which decrease the performance.

Later, we tried dynamic learing rate. We Let the model to warm up in the first 10 epoch and then Cosine Anneal. The graph of learning rate is as following:

![WechatIMG524](https://cdn.luogu.com.cn/upload/image_hosting/sbdwro2g.png)

Learning rate is a hyperparameter that determines the size of the steps taken during the optimization process of training a neural network. Dynamic learning rate make the learing process more stable and Ultimately converges to a very low loss. If we use constant learning rate, In the later stages of learning, the loss curve may oscillate and be difficult to converge.

We also introduce Batch normalization to our model. It improves the training speed, stability, and performance. Because removing Batch normalization makes our model very hard to train, **we consider Batch normalization as a defualt setting in following experiment.** It means that the model without optimization also contains Batch normalization. The experiments is as following:

|                            | Loss              | PSNR             | SSIM            |
| -------------------------- | ----------------- | ---------------- | --------------- |
| Without optimization       | 0.00226           | 26.921           | 0.770           |
| With weight decay          | 0.00205           | 27.399           | 0.774           |
| With dropout               | 0.00239           | 26.426           | 0.739           |
| With dynamic learning rate | **0.00198** | **27.493** | **0.780** |

These experiment investigate the inflution of single technique. It shows that both weight decay and dynamic learning rate help to improve the performance of the model, while only dropout decrease the performance.

### 4. Quick Start

    To run the code, you should enter the following sentences in the terminal:

```python
python train.py output_folder
#output folder is a parameter you can randomly choose
```

    After running the above code, you can get undersampling, full sampling and reconstruction images separately in different folders and a file called ***output.txt*** in which the value of loss , PSNR and SSIM are.

    Besides, you can change the parameter value setting in train.py. Below are our default setting:

```python
train(in_channels=20,
      out_channels=20,
      init_features=64,
      num_epochs=400,
      weight_decay=1e-4,
      batch_size=10,
      initial_lr=1e-4)
```
