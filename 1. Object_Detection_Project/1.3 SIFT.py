import cv2

# IMAGE LOADER 
image = cv2.imread("./image.jpeg")
image = cv2.resize(image, (400,400)) 

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# SIFT 객체 생성 
sift = cv2.SIFT_create()

# Keypoint 검출 및 특징 드스크립터 계산 
keypoints, descriptors = sift.detectAndCompute(gray , None)

# 키포인트 그리기 
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# 결과 이미지 출력 
cv2.imshow("Image with Keypoints ", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()