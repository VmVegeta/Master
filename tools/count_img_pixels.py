from PIL import Image
import numpy as np


def main(file_path: str):
    img = Image.open(file_path).convert('RGB')

    na = np.array(img)
    colors, counts = np.unique(na.reshape(-1, 3), axis=0, return_counts=1)
    colors_counts = []
    red_count = 0
    purple_count = 0
    road_count = 0
    for index, color in enumerate(colors):
        # if color[0] > 226 and 181 < color[1] < 211 and 176 < color[2] < 205:
        if color[0] > 220 and 181 < color[1] < 225 and 176 < color[2] < 212:
            red_count += counts[index]
        elif color[0] > 100 and 15 < color[1] < 75 and 50 < color[2] < 100:
            purple_count += counts[index]
        elif color[0] > 249 and 250 < color[1] and 238 < color[2] < 247:
            road_count += counts[index]
        elif color[0] > 210 and 120 < color[1] < 168 and 120 < color[2] < 170:
            purple_count += counts[index]
    """
        if color[0] > 220 and 181 < color[1] < 225 and 176 < color[2] < 212:
            red_count += counts[index]
        # 113, 27, 75
        if color[0] > 100 and 15 < color[1] < 75 and 50 < color[2] < 100:
            purple_count += counts[index]
        #colors_counts.append((color, counts[index]))
    #colors_counts.sort(key=lambda tup: tup[1])
    """
    return road_count + (red_count * 1.5) + (purple_count * 2)

def investigate(file_path: str):
    img = Image.open(file_path).convert('RGB')
    new_img = img.copy()

    for x in range(0, img.width):
        for y in range(0, img.height):
            color = img.getpixel((x, y))
            if color[0] > 220 and 181 < color[1] < 225 and 176 < color[2] < 212:
                new_img.putpixel((x, y), (0, 255, 0))
            elif color[0] > 100 and 15 < color[1] < 75 and 50 < color[2] < 100:
                new_img.putpixel((x, y), (0, 0, 255))
            elif color[0] > 249 and 250 < color[1] and 238 < color[2] < 247:
                new_img.putpixel((x, y), (255, 0, 255))
            elif color[0] > 210 and 120 < color[1] < 168 and 120 < color[2] < 170:
                new_img.putpixel((x, y), (255, 0, 0))
    new_img.show()


if __name__ == '__main__':
    investigate("C:/Users/vemun/PycharmProjects/Master/osmaug/data/tile/5_h_adt_zoom_17.jpg")