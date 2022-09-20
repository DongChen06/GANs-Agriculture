import albumentations as A
import cv2


img = 'goosegrass'  # carpetweed, crabgrass, eclipta, goosegrass

"""Rotate"""
transform1 = A.VerticalFlip(p=1)

# Read an image with OpenCV and convert it to the RGB colorspace
image1 = cv2.imread(img + ".jpg")
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Augment an image
transformed1 = transform1(image=image1)
transformed_image1 = transformed1["image"]
transformed_image1 = cv2.cvtColor(transformed_image1, cv2.COLOR_RGB2BGR)
cv2.imwrite(img + "_VerticalFlip.jpg", transformed_image1)


"""Translation"""
transform2 = A.HorizontalFlip(p=1)
image2 = cv2.imread(img + ".jpg")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# Augment an image
transformed2 = transform2(image=image2)
transformed_image2 = transformed2["image"]
transformed_image2 = cv2.cvtColor(transformed_image2, cv2.COLOR_RGB2BGR)
cv2.imwrite(img + "_HorizontalFlip.jpg", transformed_image2)


"""Brightness"""
transform3 = A.RandomBrightnessContrast(brightness_limit=0.5, p=1)
image3 = cv2.imread(img + ".jpg")
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
# Augment an image
transformed3 = transform3(image=image3)
transformed_image3 = transformed3["image"]
transformed_image3 = cv2.cvtColor(transformed_image3, cv2.COLOR_RGB2BGR)
cv2.imwrite(img + "_RandomBrightnessContrast.jpg", transformed_image3)


"""Blur"""
transform4 = A.Blur(blur_limit=30, p=1)
image4 = cv2.imread(img + ".jpg")
image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
# Augment an image
transformed4 = transform4(image=image4)
transformed_image4 = transformed4["image"]
transformed_image4 = cv2.cvtColor(transformed_image4, cv2.COLOR_RGB2BGR)
cv2.imwrite(img + "_Blur.jpg", transformed_image4)


"""GaussNoise"""
transform5 = A.GaussNoise(var_limit=(0, 100.0), mean=60, p=1)
image5 = cv2.imread(img + ".jpg")
image5 = cv2.cvtColor(image5, cv2.COLOR_BGR2RGB)
# Augment an image
transformed5 = transform5(image=image5)
transformed_image5 = transformed5["image"]
transformed_image5 = cv2.cvtColor(transformed_image5, cv2.COLOR_RGB2BGR)
cv2.imwrite(img + "_GaussNoise.jpg", transformed_image5)


"""GridDistortion"""
transform6 = A.GridDistortion(num_steps=1,p=1)
image6 = cv2.imread(img + ".jpg")
image6 = cv2.cvtColor(image6, cv2.COLOR_BGR2RGB)
# Augment an image
transformed6 = transform6(image=image6)
transformed_image6 = transformed6["image"]
transformed_image6 = cv2.cvtColor(transformed_image6, cv2.COLOR_RGB2BGR)
cv2.imwrite(img + "_GridDistortion.jpg", transformed_image6)


"""FancyPCA"""
transform7 = A.FancyPCA(alpha=0.4,p=1)
image7 = cv2.imread(img + ".jpg")
image7 = cv2.cvtColor(image7, cv2.COLOR_BGR2RGB)
# Augment an image
transformed7 = transform7(image=image7)
transformed_image7 = transformed7["image"]
# transformed_image7 = cv2.cvtColor(transformed_image7, cv2.COLOR_RGB2BGR)
cv2.imwrite(img + "_FancyPCA.jpg", transformed_image7)


"""RGBShift"""
transform8 = A.RGBShift(p=1)
image8 = cv2.imread(img + ".jpg")
image8 = cv2.cvtColor(image8, cv2.COLOR_BGR2RGB)
# Augment an image
transformed8 = transform8(image=image8)
transformed_image8 = transformed8["image"]
transformed_image8 = cv2.cvtColor(transformed_image8, cv2.COLOR_RGB2BGR)
cv2.imwrite(img + "_RGBShift.jpg", transformed_image8)