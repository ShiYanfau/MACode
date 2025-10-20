
import torch
import torch.nn as nn



class PatchEmbed(nn.Module):
    def __init__(self,img_size=224, patch_size=16, in_chans=3, embed_dim=768, lay_norm=None):
        # img_size 图像大小， patch_size 每个patch的大小， in_chans 输入通道数， embed_dim 嵌入维度， lay_norm 层归一化
        super().__init__()

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)  # (14, 14) patch网格数量
        self.num_patches = self.grid_size[0] * self.grid_size[1] #

        self.proj = nn.Conv2d(in_channels=in_chans,out_channels=embed_dim, kernel_size=patch_size, stride=patch_size) # 3x224x224 -> 768x14x14
        self.lay_norm = lay_norm(embed_dim) if lay_norm else nn.Identity() #如果有则使用，如果没有则默认保持不变

    def forward(self, x):
        B, C, H, W = x.shape # B: batch size, C: channels, H: height, W: width
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)  # B, 768, 14, 14
        x = x.flatten(2)  # B, 768, 196
        x = x.transpose(1, 2)  # B, 196, 768
        x = self.lay_norm(x)  # B, 196, 768
        return x








