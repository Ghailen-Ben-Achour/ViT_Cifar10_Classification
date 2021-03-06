# ViT_Cifar10_Classification
In this work, we focus on vision transformers for classification tasks. Transformers have been widely used is NLP and time series. We report the results of the attention mechanism to capture and extract features. The model architecture is based on [vit-pytorch](https://github.com/lucidrains/vit-pytorch). We also add the possibility to visualize the attention map.
## ViT parameters
Following the original implementation [here](https://github.com/lucidrains/vit-pytorch), we use the following parameters for each ViT model.
```python
v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
```

- `image_size`: int  
Image size. If you have rectangular images, make sure your image size is the maximum of the width and height.
- `patch_size`: int  
Number of patches. `image_size` must be divisible by `patch_size`.  
(The number of patches is: n = (image_size // patch_size) ** 2 and n **must be greater than 16**.
- `num_classes`: int  
Number of classes to classify. (outputs)
- `dim`: int  
Last dimension of output tensor after linear `transformation nn.Linear(..., dim)`.
- `depth`: int  
Number of Transformer blocks.
- `head`: int  
Number of heads in Multi-head Attention layer.
- `mlp_dim`: int  
Dimension of the MLP (FeedForward) layer.  
- `channels`: int, default `3`  
Number of image's channels.
- `dropout`: float between `[0, 1]`, default `0`.  
Dropout rate.
- `emb_dropout`: float between `[0, 1]`, default `0`.  
Number of image's channels.  
- `pool`: string  
Either `cls` token pooling or `mean` pooling.
## Usage
Install dependencies using this command:  
```bash
pip install -r requirements.txt
```
To train the ViT on image classification you can use this command:
```bash
python main.py --name trial
```

