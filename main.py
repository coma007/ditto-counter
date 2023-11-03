import sys
sys.path.append(".")

from preprocessing import *


if __name__ == "__main__":

    path = parse_args()
    images = read_all_images(path)

    counts = read_csv("ditto_count.csv")
    
    actual_values = np.zeros(len(counts))
    predicted_values = np.zeros(len(counts))

    for i, path in zip(range(len(images)), images):

        title = path.split("/")[-1]
        image = load_image(path)

        hsv_image = convert_to_hsv(image)
        purple_mask = extract_purple(hsv_image)
        _, image_bin = convert_to_bin(purple_mask)

        kernel = np.ones((3,3), np.uint8) 

        opening, closing, dilation, erosion = get_morphological_features(image_bin, kernel, 6)
        dist_transform = distance_transform(dilation) 
        sure_fg, sure_bg, unknown = extract_foreground(dist_transform, erosion, kernel, 3, 0.55)
        number_of_dittos = watershed(image, sure_fg, unknown)

        print_result(title, counts[title], number_of_dittos)
        save_result(i, actual_values, predicted_values, counts[title], number_of_dittos)

    mae = calculate_mae(predicted_values, actual_values)
    print(mae)

    
 