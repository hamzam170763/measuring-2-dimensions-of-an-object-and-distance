import cv2

class image_resizer:
    def __init__(self):
        pass
    def resize_image(self,img):
        print('Original Dimensions : ', img.shape)
        scale_percent = 15  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)

        #dimensions of resized image
        dim = (width, height)

        #Resizing
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        print('Resized Dimensions : ', resized.shape)
        #return resized image
        return resized
