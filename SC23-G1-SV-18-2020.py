

import sys
sys.path.append(".")

from preprocessing import *

if __name__ == "__main__":

    path = parse_args()
    images = read_all_images(path)

    counts = read_csv("ditto_count.csv")

    i = 0
    for path in images:
        image = load_image(path)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_purple = np.array([150, 50, 200])  # Adjust these values to match the specific shade of purple you want to detect
        upper_purple = np.array([180, 255, 255])

        purple_mask = cv2.inRange(hsv_image, lower_purple, upper_purple)
        purple_extracted = cv2.bitwise_and(image, image, mask=purple_mask)
        image = purple_extracted
        ret, image_bin = cv2.threshold(purple_mask, 1, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3,3), np.uint8) 

        opening = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel, iterations = 2) # otvaranje
        closing = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel, iterations = 2) # zatvaranje
        dilation = cv2.morphologyEx(image_bin, cv2.MORPH_DILATE, kernel, iterations = 2) # dilacija
        erosion = cv2.morphologyEx(image_bin, cv2.MORPH_ERODE, kernel, iterations = 2) # erozija
        sure_bg = cv2.dilate(closing, kernel, iterations=1)
        dist_transform = cv2.distanceTransform(dilation, cv2.DIST_L2, maskSize=5) #  DIST_L2 - Euklidsko rastojanje

        ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0) 
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]
        rgba_img = label2rgb(markers)
        unique_colours = {x for l in markers for x in l}
        number_of_dittos = len(unique_colours) - 2

        title = path.split("/")[-1]
        print(f"{title}-{counts[title]}-{number_of_dittos}")

    
 