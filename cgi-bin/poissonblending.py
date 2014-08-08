#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
import cv2


def blend(img_target, img_source, img_mask, offset=(0, 0)):
	# compute regions to be blended
	region_source = (
			max(-offset[0], 0),
			max(-offset[1], 0),
			min(img_target.shape[0]-offset[0], img_source.shape[0]),
			min(img_target.shape[1]-offset[1], img_source.shape[1]))
	region_target = (
			max(offset[0], 0),
			max(offset[1], 0),
			min(img_target.shape[0], img_source.shape[0]+offset[0]),
			min(img_target.shape[1], img_source.shape[1]+offset[1]))
	region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

	# clip and normalize mask image
	img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
	img_mask[img_mask==0] = False
	img_mask[img_mask!=False] = True

	# create coefficient matrix
	A = scipy.sparse.identity(np.prod(region_size), format='lil')
	for y in range(region_size[0]):
		for x in range(region_size[1]):
			if img_mask[y,x]:
				index = x+y*region_size[1]
				A[index, index] = 4
				if index+1 < np.prod(region_size):
					A[index, index+1] = -1
				if index-1 >= 0:
					A[index, index-1] = -1
				if index+region_size[1] < np.prod(region_size):
					A[index, index+region_size[1]] = -1
				if index-region_size[1] >= 0:
					A[index, index-region_size[1]] = -1
	A = A.tocsr()
    
	# create poisson matrix for b
	P = pyamg.gallery.poisson(img_mask.shape)

	# for each layer (ex. RGB)
	for num_layer in range(img_target.shape[2]):
		# get subimages
		t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
		s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
		t = t.flatten()
		s = s.flatten()

		# create b
		b = P * s
		for y in range(region_size[0]):
			for x in range(region_size[1]):
				if not img_mask[y,x]:
					index = x+y*region_size[1]
					b[index] = t[index]

		# solve Ax = b
		x = pyamg.solve(A,b,verb=False,tol=1e-10)

		# assign x to target image
		x = np.reshape(x, region_size)
		x[x>255] = 255
		x[x<0] = 0
		x = np.array(x, img_target.dtype)
		img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

	return img_target


def test():

	image = np.asarray(PIL.Image.open("../gousei/1.jpg"))
	image.flags.writeable = True
	image2 = np.asarray(PIL.Image.open("../gousei/c.jpg"))
	image2.flags.writeable = True
	
	image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
	
	#res2 = cv2.addWeighted(image2, 0.5, image, 0.5, 0)
	# —¼•û‹¤’Ê‚Ìƒ}ƒXƒN
	res = cv2.bitwise_and(image, image2)
	
	cv2.imwrite("../gousei/kyoutu.jpg", res)
	
	face1 = np.asarray(PIL.Image.open("../gousei/3.jpg"))
	face1.flags.writeable = True
	face2 = np.asarray(PIL.Image.open("../gousei/d.jpg"))
	face2.flags.writeable = True

	face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
	
	ans = blend(face1, face2, res)
	ans = PIL.Image.fromarray(np.uint8(ans))
	ans.save("../gousei/ans.jpg")
	
	return ans

def tmp() :
	img = cv2.imread("../gousei/ans.jpg")
	cv2.imshow("poisson", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
#def paste(im)

if __name__ == '__main__':
	im = test()
	print "Finish!"
	
	image = cv2.imread("../gousei/2.jpg")
	image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
	cascade = cv2.CascadeClassifier("../../haarcascade_frontalface_alt_tree.xml")
	
	facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
	a = cv2.imread("../gousei/3.jpg")
	print a.shape
	if len(facerect) > 0 :
		for face in facerect :
			print face[0], "  ", face[1], "  ", face[2], "  ", face[3]
			f = PIL.Image.open("../gousei/2.jpg")
			f.paste(im, (face[0], face[1]))
			f.save("../gousei/gousei1.jpg")
	
#	paste(im)
#	tmp()
