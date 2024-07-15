import cv2
import numpy as np

#Load the image
image_path = 'C:/Users/sk100/Documents/Sakshi/edge_contour_detection/images/input.png'
image = cv2.imread(image_path)

#Check if the image is loaded
if image is None:
    print("Error: Could not read the image")
else:
    print("Image loaded successfully")

#As edge detection works better with gray scale images therefore converting it

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Display the gray image
cv2.imshow("Grayscale Image",gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Performing the edge detection by canny edge detection
edges = cv2.Canny(gray_image,threshold1=100, threshold2=200)

#Display the edge detected image
cv2.imshow("Edge Detected Image", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Next step is to find contours
contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Draw contours on the original image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

#Display the contour image
cv2.imshow("Contours", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Save the image in the output folder

cv2.imwrite('C:/Users/sk100/Documents/Sakshi/edge_contour_detection/output/output.png', edges)
cv2.imwrite('C:/Users/sk100/Documents/Sakshi/edge_contour_detection/output/contour_image.png', contour_image)
cv2.imwrite('C:/Users/sk100/Documents/Sakshi/edge_contour_detection/output/grayscale.png',gray_image)

