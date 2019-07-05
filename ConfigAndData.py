from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()


classLabelNames = ['T-shirt/top(T恤)', 'Trouser(裤子)', 'Pullover(套衫)', 'Dress(裙子)', 'Coat(外套)',
                   'Sandal(凉鞋)', 'Shirt(衬衫)', 'Sneaker(运动鞋)', 'Bag(包)', 'Ankle boot(踝靴)']


inputShape = (28, 28, 1)
