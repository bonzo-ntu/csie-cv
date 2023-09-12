# %%
import os
# os.chdir('/home/jupyter/ntu/csie-cv/hw1')

from PIL import Image
import numpy as np
import copy

def save_image(img, path='./lena.bmp'):
    img_ = Image.fromarray(np.array(img, dtype='uint8'), mode='L')
    img_.save(path)
    return img_

# %%
img = Image.open('./lena.bmp')
img

# %%
img_array = np.array(img)
width, height = img_array.shape
img_list = img_array.tolist()

# %% [markdown]
# # Part I
# ## a. upside down

# %% [markdown]
# ### method 1: row exchange

# %%
def upside_down(img, height=height, width=width):
    for y in range(height//2):
        for x in range(width):
            elm = img[y][x]
            img[y][x] = img[(height-1)-y][x]
            img[(height-1)-y][x] = elm
    return img
    
result = copy.deepcopy(img_list)
save_image(upside_down(result), './lena_upside_down.bmp')

# %% [markdown]
# ### method 2: reverse y index

# %%
# result = img_array.copy()
# result = result[np.arange(511,-1,-1)]

# save_image(result, './lena_upside_down.bmp')

# %% [markdown]
# ## b. right-side-left

# %% [markdown]
# ### method 1: col exchange

# %%
def rightside_left(img, height=height, width=width):
    for y in range(height):
        for x in range(width//2):
            elm = img[y][x]
            img[y][x] = img[y][(width-1)-x]
            img[y][(width-1)-x] = elm
    return img

result = copy.deepcopy(img_list)
save_image(rightside_left(result), './lena_rightside_left.bmp')

# %% [markdown]
# ### method 2: reverse x index

# %%
# result = img_array.copy()
# result = result[:,range(511,-1,-1)]

# save_image(result, './lena_rightside_left.bmp')

# %% [markdown]
# ## c. diagonally flip

# %% [markdown]
# ### method 1: row exanch + col exchange

# %%
def digonal_flip(img, height=height, width=width):
    img = upside_down(img, height, width)
    img = rightside_left(img, height, width)
    return img
    
result = copy.deepcopy(img_list)
save_image(digonal_flip(result), './lena_diagonal_mirrored.bmp')

# %% [markdown]
# ### method 2: reverse y index + reverse x index

# %%
# result = img_array.copy()
# result = result[np.arange(511,-1,-1)]
# result = result[:,np.arange(511,-1,-1)]

# save_image(result, './lena_diagonal_mirrored.bmp')

# %% [markdown]
# # 3. shrink the size to half

# %% [markdown]
# ### method a: interleaving drop

# %%
def shrink(img, height=height, width=width, scale=2):
    for y in range(0, height, scale):
        for x in range(0, width, scale):
            elm = img[y][x]
            img[y//scale][x//scale] = elm
    img = [ [img[y][x] for x in range(0, width//scale)] for y in range(0, height//scale)]
    return img

result = copy.deepcopy(img_list)
save_image(shrink(result, scale=2), './lena_shrink.bmp')

# %% [markdown]
# ### method b: mean

# %%
# def shrink_mean(img, height=height, width=width):
#     for y in range(0, height, 2):
#         for x in range(0, width, 2):
#             elm = sum((img[y][x], img[y][x+1], img[y+1][x],img[y+1][x+1]))/4
#             img[y//2][x//2] = elm
#     img = [ [img[y][x] for x in range(0, width//2)] for y in range(0, height//2)]
#     return img

# result = copy.deepcopy(img_list)
# save_image(shrink_mean(result), './lena_shrink_mean.bmp')

# %% [markdown]
# # 4. Binarize 

# %% [markdown]
# ### method a: sequensial

# %%
def binarize(img, height=height, width=width):
    for y in range(height):
        for x in range(width):
            img[y][x] = 255 if img[y][x] >= 128 else 0
    return img
    
result = copy.deepcopy(img_list)
save_image(binarize(result), './lena_binarize.bmp')

# %% [markdown]
# ### method b: parallel

# %%
# result = img_array.copy()
# bright = result >= 128
# result[bright] = 255
# result[~bright] = 0

# save_image(result, './lena_binarize.bmp')

# %% [markdown]
# # 6. Negative

# %% [markdown]
# ### method a: sequensial

# %%
# def negative(img, height=height, width=width):
#     for y in range(height):
#         for x in range(width):
#             img[y][x] = 255-img[y][x]
#     return img

# result = copy.deepcopy(img_list)
# save_image(negative(result), './lena_negative.bmp')

# %%



