import torch
from pytorch_pretrained_vit import ViT


class ViTWrapper(torch.nn.Module):

    def __init__(
            self, 
            src:str="B_16_imagenet1k",
            pretrained:bool=True,
            **kwargs
            ):
        super().__init__()
        # Remove pretrained from kwargs if present to avoid duplicate
        kwargs.pop('pretrained', None)
        self.model = ViT(src, pretrained=pretrained, **kwargs)

    def forward(self, x):
        return self.model(x)
    

if __name__ == "__main__":
    model = ViTWrapper()
    breakpoint()

