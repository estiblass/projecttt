import os
import cv2



def cut_rows_in_image(image_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to convert to binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours of lines in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from top to bottom
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Group contours into rows
    rows = []
    current_row = []
    previous_y = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if previous_y is None:
            current_row.append(contour)
        elif y - previous_y <= h:
            current_row.append(contour)
        else:
            rows.append(current_row)
            current_row = [contour]

        previous_y = y

    rows.append(current_row)
    # Crop and save each valid row as a separate image
    for i, row in enumerate(rows):
        # Calculate the bounding box of the row
        x_min = min(cv2.boundingRect(contour)[0] for contour in row)
        x_max = max(cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in row)
        y_min = min(cv2.boundingRect(contour)[1] for contour in row)
        y_max = max(cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] for contour in row)

        # Check if the row meets the minimum width and aspect ratio thresholds
        if x_max - x_min >= 20 and (y_max - y_min) / (x_max - x_min) <= 0.4:  # Adjust the thresholds as needed
            # Crop the row from the image
            row_image = image[y_min:y_max, x_min:x_max]

            # Save the row image to the output folder
            row_path = os.path.join(output_folder, f"row_{i + 1}.png")
            cv2.imwrite(row_path, row_image)

            print(f"Saved row {i + 1} as {row_path}")


# Example usage
image_path = "p.jpg"
output_folder = "rows"
cut_rows_in_image(image_path, output_folder)