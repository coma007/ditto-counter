import sys
sys.path.append(".")

from preprocessing import *


if __name__ == "__main__":

    path = parse_args()
    images = read_all_images(path)

    counts = read_csv("ditto_count.csv")
    
    actual_values = np.zeros(len(counts))
    predicted_values = np.zeros(len(counts))


if __name__ == "__main__":

    path = parse_args()
    images = read_all_images(path)

    counts = read_csv("ditto_count.csv")
    
    actual_values = np.zeros(len(counts))
    predicted_values = np.zeros(len(counts))

    for i, path in zip(range(len(images)), images):

        title = path.split("/")[-1]
        image = load_image_hsv(path=path)

        purple_mask = extract_purple(hsv_image=image)
        _, image_bin = convert_to_bin(image=purple_mask)

        kernel = np.ones((3,3), np.uint8) 

        opening, closing, dilation, erosion = get_morphological_features(image=image_bin, kernel=kernel, iter=6)
        dist_transform = distance_transform(feature=dilation) 
        sure_fg, sure_bg, unknown = extract_foreground(dist_transform=dist_transform, background_feature=closing, kernel=kernel, iter=3, percentage=0.55)
        number_of_dittos = watershed(image=image, sure_fg=sure_fg, unknown=unknown)

        print_result(title, counts[title], number_of_dittos)
        save_result(index=i, actual_values_array=actual_values, predicted_values_array=predicted_values, actual_value=counts[title], predicted_value=number_of_dittos)

    mae = calculate_mae(predicted_values, actual_values)
    print(mae)

    
 