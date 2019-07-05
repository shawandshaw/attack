from main import aiTest
from ConfigAndData import test_images, inputShape

test_images = test_images[0:100]

test_images = test_images.reshape(len(test_images), *inputShape)

generateImages = aiTest(test_images, (len(test_images), 28, 28, 1))
print(generateImages.shape)
