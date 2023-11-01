

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

        image_blur = cv2.GaussianBlur(rgb_to_grayscale(image), (5, 5), 0)
        image_bin =  cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # ret, image_bin = cv2.threshold(rgb_to_grayscale(image), 0, 255, cv2.cv2.THRESH_OTSU)
        # image_bin = 255 - image_bin
        # image_bin = grayscale_to_bin(rgb_to_grayscale(image))
        
        kernel = np.ones((3,3), np.uint8) 

        opening = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel, iterations = 3) # otvaranje
        closing = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel, iterations = 3) # zatvaranje
        dilation = cv2.morphologyEx(image_bin, cv2.MORPH_DILATE, kernel, iterations = 3) # dilacija
        erosion = cv2.morphologyEx(image_bin, cv2.MORPH_ERODE, kernel, iterations = 3) # erozija
        sure_bg = cv2.dilate(closing, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, maskSize=0) #  DIST_L2 - Euklidsko rastojanje

        ret, sure_fg = cv2.threshold(dist_transform, 0.65 * dist_transform.max(), 255, 0) 
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

    
 