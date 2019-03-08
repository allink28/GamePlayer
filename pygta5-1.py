import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D


# WIDTH = 800
# HEIGHT = 640
# gta_vertices = np.array([[10,500],[10,300],[300,200],[500,200],[WIDTH,300],[WIDTH,500]], np.int32)
MC_WIDTH = 856
MC_HEIGHT = 482
WIDTH = MC_WIDTH
HEIGHT = MC_HEIGHT
mc_vertices = np.array([[0,0],[0,433],[561,433],[561,338],[WIDTH,338],[WIDTH,0]], np.int32)


def roi(img, vertices):
    """
    Region of Interest
    :param vertices: Vertices that define the region of interest
    :return: Masked image
    """
    # blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    """
    Edge detection!
    :return: A grayscale image with edges detected
    """
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=300)

    processed_img = roi(processed_img, [mc_vertices])

    return processed_img


def screen_record():
    last_time = time.time()
    while True:
        # 800x600 windowed mode
        screen = np.array(ImageGrab.grab(bbox=(0,32, WIDTH,HEIGHT)))
        # print('loop took {} seconds'.format(time.time()-last_time))
        # last_time = time.time()
        new_screen = process_img(screen)
        cv2.imshow('window', new_screen)
        # cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def countdown(count):
    # Countdown so you have time to switch to game
    for i in list(range(count))[::-1]:
        print(i + 1)
        time.sleep(1)


if __name__ == "__main__":
    # countdown(4)
    screen_record()
