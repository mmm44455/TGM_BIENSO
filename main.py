import cv2
import pytesseract
from ultralytics import YOLO
import os
import easyocr
import re
# Load a model
model = YOLO(r"D:\Thị giác máy\BTL-plane\runs\detect\train\weights\best.pt")  # Load a pretrained model
results = model(r"D:\Thị giác máy\BTL-plane\demo.jpg", show=True)

# Assuming there is only one image in the batch
image_index = 0
image_info = results[image_index]

# Retrieve bounding boxes
boxes = image_info.boxes.data.cpu().numpy()

# Load the image
image_path = r"D:\Thị giác máy\BTL-plane\demo.jpg"
image = cv2.imread(image_path)

# Create a folder to save cropped images
output_folder = "cropped_images"
os.makedirs(output_folder, exist_ok=True)

# Process each bounding box
for i, box in enumerate(boxes):
    x_min, y_min, x_max, y_max = map(int, box[:4])

    # Crop the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Save the cropped image
    output_path = os.path.join(output_folder, f"cropped_image_{i + 1}.png")
    cv2.imwrite(output_path, cropped_image)
    # Show the cropped image
    cv2.imshow(f"anh cat ", cropped_image)

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # text = pytesseract.image_to_string(gray_image)
    # # # Làm sạch văn bản (chỉ giữ lại chữ và số)
    # cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    # # In kết quả sau khi làm sạch
    # print("Cleaned text:", cleaned_text)

    _, binary_image = cv2.threshold(gray_image, 94, 255, cv2.THRESH_BINARY)
    # # cv2.imshow(" ",binary_image)
    # #
    reader = easyocr.Reader(['en'])
    result_easyocr = reader.readtext(gray_image, detail=0, paragraph=True, contrast_ths=1.5, adjust_contrast=0.5)
    alphanumeric_easyocr = ''.join(re.findall(r'[a-zA-Z0-9]', ''.join(result_easyocr)))
    print("EasyOCR Result:", alphanumeric_easyocr)

    # threh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # contours, hierarchy = cv2.findContours(threh,1,2)
    # largest_contour =[0,0]
    # for cnt in contours:
    #     leght= 0.01 * cv2.arcLength(cnt,True)
    #     approx = cv2.approxPolyDP(cnt,leght,True)
    #     if len(approx)==4:
    #         area = cv2.contourArea(cnt)
    #         if area>largest_contour[0]:
    #             largest_contour =[cv2.contourArea(cnt),cnt,approx]
    # x,y,w,h = cv2.boundingRect(largest_contour[1])
    # image=cropped_image[y:y+h, x:x+w]
    # cv2.drawContours(image,[largest_contour[1]],0,(0,255,0),8)
    # img2=cropped_image[y:y+h, x:x+w]
    # cv2.drawContours(img2,[largest_contour[1]],0,(0,255,0),8)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(3,3),0)
    # threh = cv2.adaptiveThreshold(blur,0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # opening = cv2.morphologyEx(threh, cv2.MORPH_OPEN, kernel,iterations=1)
    # invert = 255 - opening
    # data = pytesseract.image_to_string(invert,lang='eng',config='--psm 6')
    # print(data)
cv2.waitKey(0)
cv2.destroyAllWindows()