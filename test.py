#%%
import torch

#%%
B, C, X, Y, Z = 4, 3, 17, 17, 17
sig = torch.randn((B, C, X, Y, Z))
x_ft = torch.fft.rfftn(sig, dim=[2,3,4])
# %%
m1, m2, m3 = 12, 12, 12
w1 = torch.rand(C, C, m1, m2, m3, dtype=torch.cfloat)
# %%
x_ft.shape
# %%
out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat)
out_ft[..., :5] = x_ft[..., :5]
output = torch.fft.irfftn(out_ft, s=[X, Y, Z], dim=[2, 3, 4])
out2 = torch.fft.irfftn(x_ft[..., :5], s=[X, Y, Z], dim=[2, 3, 4])
# %%
print(output - out2)
# %%
out2.shape
# %%
