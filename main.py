# import matplotlib.pyplot as plt
import numpy as np
import tensorflow.python.keras as keras
import tensorflow.python.keras.api._v1.keras.backend as K
# from skimage.measure import compare_ssim as ssim

# from ConfigAndData import (classLabelNames)

epochs = 1

# input image dimensions


def hack_one_image(model, org_img):
    input_layer = model.layers[0].input
    output_layer = model.layers[-1].output

    # 创建图像变量
    org_img = np.expand_dims(org_img, axis=0)
    hack_img = np.copy(org_img)

    # 设定阈值保证相似度
    max_change = 0.02
    max_change_above = org_img + max_change
    max_change_below = org_img - max_change
    # 攻击前模型预测的label
    actual_label = np.argmax(model.predict(org_img))
    # 获取损失函数，梯度函数，function实例
    cost_function = output_layer[0, actual_label]
    gradient_function = K.gradients(cost_function, input_layer)[0]
    instance = K.function([input_layer, K.learning_phase()],
                          [cost_function, gradient_function])

    # 生成攻击图像
    learning_rate = 0.001
    count = 0
    step = 0.005
    while np.argmax(model.predict(hack_img)) == actual_label:
        count += 1
        if count % 50 == 0:
            max_change_below -= step
            max_change_above += step
        cost, gradient = instance([hack_img, 0])
        n = np.sign(gradient)
        hack_img -= n * learning_rate
        hack_img = np.clip(hack_img, max_change_below, max_change_above)
        hack_img = np.clip(hack_img, 0, 1.0)

    hack_label = np.argmax(model.predict(hack_img))
    # plt.imshow(org_img.reshape(28, 28))
    # plt.title(classLabelNames[actual_label])
    # plt.show()
    # plt.imshow(hack_img.reshape(28, 28))
    # plt.title(classLabelNames[np.argmax(model.predict(hack_img))])
    # plt.show()
    # sm = ssim(org_img.squeeze(0), hack_img.squeeze(0), multichannel=True)
    print('orgImgLabel: {}, hackImgLabel: {}'.format(actual_label, hack_label))
    hack_img = hack_img.squeeze(0)
    hack_img *= 255.0
    hack_img = hack_img.astype('uint8')
    return hack_img


def aiTest(images, shape):
    model = keras.models.load_model('myModel.h5')
    generate_images = np.array([], dtype='uint8')
    cnt = 0
    for img in images:
        print('hacking img {}......'.format(cnt))
        cnt += 1
        hack_img = hack_one_image(model, img / 255.0)
        generate_images = np.append(generate_images, hack_img)

    generate_images = generate_images.reshape(shape)
    return generate_images
