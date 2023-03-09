import os
import cv2 as cv

 
def ImgFill(img):

    contours, hierarch = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])   #计算轮廓所占面积
        if area < 1:                   #轮廓面积，可以自己随便调
            cv.drawContours(img,[contours[i]],0,0,-1)         #该轮廓区域填0
        else:
            cv.drawContours(img,[contours[i]],0,255,-1)
        
    return img

def get_data(data_src, data_dest = 'dataset'):

    if not os.path.exists(data_dest):
        os.mkdir(data_dest)
    if not os.path.exists(data_dest+'/inputs'):
        os.mkdir(data_dest+'/inputs')
    if not os.path.exists(data_dest+'/labels'): 
        os.mkdir(data_dest+'/labels')

    path_images = []
    path_labels = []

    patient_list = os.listdir(data_src)
    for patient in patient_list:
        info = os.listdir(os.path.join(data_src, patient))
        images = os.listdir(os.path.join(data_src, patient, info[0]))
        labels = os.listdir(os.path.join(data_src, patient, info[1]))
        for image, label in zip(images, labels):
            path_images.append(os.path.join(data_src, patient, info[0], image))
            path_labels.append(os.path.join(data_src, patient, info[1], label))

    for i, (image, label) in enumerate(zip(path_images, path_labels)):
        if i % 30 == 0 and i != 780 and i != 930:
            image_origin = cv.imread(image)[:,1:,::-1]
            label_origin = cv.imread(label)[:,1:,::-1]
        else:
            image_origin = cv.imread(image)[:,:,::-1]
            label_origin = cv.imread(label)[:,:,::-1]
        
        label_final = cv.subtract(label_origin, image_origin)
        label_final = cv.cvtColor(label_final, cv.COLOR_BGR2GRAY)
        label_final = ImgFill(label_final)
        image_origin = cv.cvtColor(image_origin, cv.COLOR_BGR2GRAY)

        cv.imwrite(data_dest + f'/inputs/img_{i+1}.jpg', image_origin)
        cv.imwrite(data_dest + f'/labels/img_{i+1}.jpg', label_final)