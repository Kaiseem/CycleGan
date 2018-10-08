from scipy.misc import imread
import scipy
import copy,os
import numpy as np
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, fine_size=256):
    img = imread(image_path, mode='RGB').astype(np.float)
    img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = img / 127.5 - 1.
    return img

def xingkong():
    img_path = 'G:/xingkong.jpg'
    img = imread(img_path, mode='RGB').astype(np.float)
    h1 = int(np.ceil(np.random.uniform(1e-2, 1080 - 512)))
    w1 = int(np.ceil(np.random.uniform(1e-2, 863 - 512)))
    img = img[h1:h1 + 512, w1:w1 + 512]
    return scipy.misc.imresize(img, [256, 256])
def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False,star=False):
    if star==False:
        img_A = scipy.misc.imread(image_path[0], mode='RGB').astype(np.float)
        img_B = scipy.misc.imread(image_path[1], mode='RGB').astype(np.float)
        if not is_testing:
            img_A = scipy.misc.imresize(img_A, [load_size, load_size])
            img_B = scipy.misc.imresize(img_B, [load_size, load_size])
            h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
            w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
            img_A = img_A[h1:h1 + fine_size, w1:w1 + fine_size]
            img_B = img_B[h1:h1 + fine_size, w1:w1 + fine_size]
            if np.random.random() > 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
        else:
            img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
            img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = xingkong()
        img_B = scipy.misc.imread(image_path[1], mode='RGB').astype(np.float)
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        img_B = img_B[h1:h1 + fine_size, w1:w1 + fine_size]
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def save_images(images, size, image_path):
    images=(images+1.)/2.
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return scipy.misc.imsave(image_path,img)
def load_train(rootdir):
    filelist=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        if '.jpg' in os.path.basename(list[i]).lower():
            filelist.append(rootdir+list[i])
    return filelist

#sample_image = np.expand_dims(load_test_data('G:/cyclegan/horse2zebra/trainA/n02381460_2.jpg'), axis=0)
#save_images(sample_image, [1, 1],  'G:/1.jpg')