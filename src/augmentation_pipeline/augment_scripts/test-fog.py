from fog import add_fog
import cv2
import os

src = 'data/raw/visdrone/sample.jpg'
dst = 'data/augmented/fog/fogged.jpg'

img = cv2.imread(src)
fogged = add_fog(img, fog_intensity=0.6, depth_blend=0.4)
cv2.imwrite(dst, fogged)
