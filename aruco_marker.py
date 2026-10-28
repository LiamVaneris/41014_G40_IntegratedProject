import cv2
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(dictionary, 0, 400)
marker = cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
cv2.imwrite("assets/aruco_0.png", cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR))
print("Saved assets/aruco_0.png")
