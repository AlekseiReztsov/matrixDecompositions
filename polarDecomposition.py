from PIL import Image
from numpy import asarray

image = Image.open('data/mario.jpg').convert('L')
image.show()

data = asarray(image)
print(type(data))
print(data.shape)