import cv2

def show_image_from_path(img_path):
    img = cv2.imread(img_path)
    cv2.imshow("image", img)
    cv2.waitKey(0)

def save_image_to_path(img, img_path):
    cv2.imwrite(img_path, img)