import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, W, A, S, D


def process_img(original_image):
    """
    Edge detection!
    :return: A grayscale image with edges detected
    """
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img


def screen_record():
    last_time = time.time()
    while True:
        # 800x600 windowed mode
        screen = np.array(ImageGrab.grab(bbox=(0,40, 800,640)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen = process_img(screen)
        cv2.imshow('window', new_screen)
        # cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    screen_record()
