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
## Usage
- `image_size`: int
