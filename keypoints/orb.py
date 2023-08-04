import cv2
import numpy as np

img1_path = "/Users/cyan/Dataset/NCL/H&E_IMC/Pair/002_00/HE002.tif"
img2_path = "/Users/cyan/Dataset/NCL/H&E_IMC/Pair/002_00/002_panorama.tif"

# Load the two images
image1 = cv2.imread(img1_path)
image2 = cv2.imread(img2_path)


image1 = cv2.resize(image1, np.array(image1.shape)[:2][::-1] // 4)
image2 = cv2.resize(image2, np.array(image2.shape)[:2][::-1] // 4)


# r, c, _ = image1.shape
# image2 = image1[r // 3 * 2 : r // 4 * 3, c // 3 : c // 2, :]
# image1 = image1[r // 2 :, : c // 2, :]

# height, width = image2.shape[:2]

# angle = 30
# center = (width // 2, height // 2)

# # Compute the rotation matrix
# rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# # Apply the rotation to the image using warpAffine
# image2 = cv2.warpAffine(image2, rotation_matrix, (width, height))


# Convert the images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# gray_image1 = cv2.resize(gray_image1, np.array(gray_image1.shape)[::-1] // 3)
# gray_image2 = cv2.resize(gray_image2, np.array(gray_image2.shape)[::-1])


# source_points = np.float32([[50, 50], [200, 50], [50, 200]])
# destination_points = np.float32([[10, 100], [150, 50], [100, 250]])

# # Compute the affine transformation matrix
# affine_matrix = cv2.getAffineTransform(source_points, destination_points)

# # Apply the affine transformation to the image
# gray_image2 = cv2.warpAffine(gray_image2, affine_matrix, (width, height))


# Create ORB detector
orb = cv2.ORB_create(nfeatures=300)

# Detect keypoints and compute descriptors for the images
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)


# img_pts_1 = cv2.cvtColor(gray_image1, cv2.COLOR_GRAY2BGR)
# img_pts_2 = cv2.cvtColor(gray_image2, cv2.COLOR_GRAY2BGR)
img_pts_1 = image1
img_pts_2 = image2

for kp in keypoints1:
    center = np.array(kp.pt).astype(np.int32)
    img_pts_1 = cv2.circle(img_pts_1, center, 5, (0, 255, 0), 2)

for kp in keypoints2:
    center = np.array(kp.pt).astype(np.int32)
    img_pts_2 = cv2.circle(img_pts_2, center, 5, (0, 255, 0), 2)


cv2.imwrite("outputs/tmp/pt1.png", img_pts_1)
cv2.imwrite("outputs/tmp/pt2.png", img_pts_2)
# exit()

# Create a Brute-Force Matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Perform matching
matches = bf.match(descriptors2, descriptors1)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 10 matches
matching_result = cv2.drawMatches(
    image2,
    keypoints2,
    image1,
    keypoints1,
    matches[:10],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv2.imwrite("outputs/tmp/orb_matching_patch_rotate_scale_affine.png", matching_result)
# Display the matching result
# cv2.imshow("Matching Result", matching_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
