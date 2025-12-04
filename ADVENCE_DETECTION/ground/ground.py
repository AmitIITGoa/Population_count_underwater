import cv2

def draw_yolo_box(image_path, yolo_line):
    # 1. Load the image
    img = cv2.imread(image_path)
    
    # Get image dimensions (Height and Width)
    h_img, w_img, _ = img.shape

    # 2. Parse the coordinate string
    parts = yolo_line.split()
    class_id = int(parts[0])
    x_center_norm = float(parts[1])
    y_center_norm = float(parts[2])
    width_norm = float(parts[3])
    height_norm = float(parts[4])

    # 3. Convert Normalized Coordinates to Pixel Coordinates
    # Calculate the center in pixels
    x_center = int(x_center_norm * w_img)
    y_center = int(y_center_norm * h_img)
    
    # Calculate width and height in pixels
    box_w = int(width_norm * w_img)
    box_h = int(height_norm * h_img)

    # 4. Calculate Top-Left Corner (x1, y1) and Bottom-Right Corner (x2, y2)
    # We subtract half the width/height from the center to find the top-left
    x1 = int(x_center - (box_w / 2))
    y1 = int(y_center - (box_h / 2))
    x2 = x1 + box_w
    y2 = y1 + box_h

    # 5. Draw the Rectangle
    # Color is BGR (Blue, Green, Red). Let's use Red (0, 0, 255)
    # Thickness is 2
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Optional: Put text ID
    cv2.putText(img, f"ID: {class_id}", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 6. Show or Save the result
    # cv2.imshow("Result", img)
    cv2.imwrite("result_with_box.jpg", img) # Saves the file
    print("Image saved as 'result_with_box.jpg'")
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# --- Run the function ---
# Replace 'image_04156b.jpg' with the actual path to your downloaded image
coordinate_string = "0 0.9296875 0.6587962962962963 0.140625 0.1675925925925926"
draw_yolo_box('image.png', coordinate_string)