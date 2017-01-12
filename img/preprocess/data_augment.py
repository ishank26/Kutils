import numpy as np
import menpo.io as mo
from menpo.transform import Translation

np.random.seed(1707)

# rotate

def get_data(data_path):
    data = []
    for i in data_path:
        img = mo.import_image(i)
        data.append(img)
    return data


def rotate(img_data):
    rotated = []
    rotation = [-15.0, 15.0]
    for i in range(len(img_data)):
        img = img_data[i]
        j = np.random.randint(2)
        rot = img.rotate_ccw_about_centre(
            rotation[j], degrees=True, retain_shape=True)
        rotated.append(rot)
    return rotated


# translate

def translate(img_data):
    translation = [[30, 30], [30, -30], [-30, 30], [-30, -30]]
    translated = []
    for i in range(len(img_data)):
        j = np.random.randint(4)
        img = img_data[i]
        shift = Translation(translation[j], skip_checks=False)
        trans = img.warp_to_shape((224, 224), shift, warp_landmarks=False)
        translated.append(trans)
    return translated


def out_img(list1, list2, type):
    for j in range(len(list2)):
        mo.export_image(list2[j], "rot/rot_" + str(type) +
                        str(j) + ".jpg", overwrite=True)

    for i in range(len(list1)):
        mo.export_image(list1[i], "trans/trans_" +
                        str(type) + str(i) + ".jpg", overwrite=True)


''''
import skimage.transform as t 
import numpy as np
from skimage import io

img0=io.imread("/home/neo/work/cnn_down/menpo_script/resize/resz0.jpg")
center_shift = np.array((224, 224)) / 2. - 0.5

translation=(30,30)
rotation=30
tform_center = t.SimilarityTransform(translation=translation)
tform_augment = t.AffineTransform(rotation=np.deg2rad(rotation))

img=t.warp(img0,tform_center,output_shape=(224,224))

#io.imshow(img)
#io.show()
#io.imsave("translated.png",img)'''
