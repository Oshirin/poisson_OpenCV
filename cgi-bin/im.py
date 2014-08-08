#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cgi
import os, sys
import cv2
import time
import codecs
import numpy as np
import scipy.sparse
import PIL.Image
import pyamg


print "Content-Type: text/html; charset=utf-8;\r\n"

print "<html><body>"


# 画像保存(upload_dir : アップロードディレクトリ)
def save_upload_file (upload_dir, fileitem1, fileitem2, i = 0):
	
	# 画像があるか判定
	if form.has_key("img1") and form.has_key("img2") :
		form_ok = 1
	if form_ok == 0 :
		print "<h1>ERROR</h1>"
	
	# ファイルのパスを設定(画像の名前は、そのときの時間+ファイル名+拡張子)
	# str() : 数字を文字列に変換
	# os.path.basename( path(ファイルパスのこと) ) : パスpathの末尾のファイル名を返す(今回の場合、画像名.拡張子)
	Path1 = os.path.join(upload_dir,str(int(time.time())) + str(i) + os.path.basename(fileitem1.filename))
	i += 1
	
	# ファイルのアップロード
	# ファイル書き込み
	fout = file(Path1, 'wb')
	# アップロードされたデータから100000バイト読み込み
	while 1:
		chunk = fileitem1.file.read(100000)
		if not chunk: break # 読み込むデータが無くなればループを抜ける
		fout.write(chunk)
	fout.close()
	
	# img2 のための保存パス
	Path2 = os.path.join(upload_dir,str(int(time.time())) + str(i) + os.path.basename(fileitem2.filename))
	
	fout = file(Path2, 'wb')
	while 1:
		chunk = fileitem2.file.read(100000)
		if not chunk: break
		fout.write(chunk)
	fout.close()
	
	print("セーブ完了!")
	# 保存した画像のパスを返す
	return Path1, Path2
	


# 計算量を落とす
def unique(a):
	""" remove duplicate columns and rows
		from http://stackoverflow.com/questions/8560440 """
	order = np.lexsort(a.T)
	a = a[order]
	diff = np.diff(a, axis=0)
	ui = np.ones(len(a), 'bool')
	ui[1:] = (diff != 0).any(axis=1)
	return a[ui]

def skin_detection(image, faces, Path, flood_diff=3, min_face_size=(30,30), num_iter=3, verbose=False, step=1):

	# マスク処理した後の画像保存path
	IMAGE_UP = "./cgi-bin/image/" +  os.path.basename(Path)

	image_original = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	image2 = np.copy(image_original)
	image2[image2==[0,0,0]] = 1

	# 抽出する色を設定
	skin_color = np.zeros((len(faces), 3))
	for i, face in enumerate(faces):
		image_face = image2[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
		skin_color[i] = np.array([image_face[image_face.shape[0]/2, image_face.shape[1]/2]])

	mask = np.zeros(image_original.shape)

	for i in range(num_iter):
		for face in faces:
			# for each pixel, call floodFill(image2) if it is close to skin_color
			for y in range(0, image2[face[1]:face[1]+face[3],face[0]:face[0]+face[2]].shape[0], step):
				if verbose:
					print 'iter: %d, y:%d' % (i, y)
				for x in range(0, image2[face[1]:face[1]+face[3],face[0]:face[0]+face[2]].shape[1], step):
					color = image2[face[1]:face[1]+face[3],face[0]:face[0]+face[2]][y,x]
					if (color!=(0,0,0)).any():
						if any((np.abs(skin_color-color)<=(flood_diff,)*3).all(1)):
							cv2.floodFill(image2[face[1]:face[1]+face[3],face[0]:face[0]+face[2]], None, (x,y), (0, 0, 0), loDiff=(flood_diff,)*3, upDiff=(flood_diff,)*3)

		# update mask image and skin_color
		mask[image2==(0,0,0)] = 255
		skin_color = image_original[mask.nonzero()]
		skin_color.shape = (skin_color.shape[0]/3, 3)
		skin_color = unique(skin_color)
	
	cv2.imwrite(IMAGE_UP, mask)
	im = np.copy(image)
	im_mask = cv2.imread(IMAGE_UP)
	res = cv2.bitwise_and(im, im_mask, im)
	os.remove(IMAGE_UP)
	FACE_UP = "./cgi-bin/face/kokoko" + os.path.basename(Path)
	#mask = np.bool_(mask)
	
	# マスクのパス、肌色部分だけの画像を返す
	return res, mask



# cv_mask1(顔画像, グレースケール)
def cv_mask1(face_img, face_gray, image, face) :

	face_im = np.zeros(face_gray.shape)
	# ガウシアンフィルタ
	im_gray = cv2.GaussianBlur(face_gray, (9, 9), 0)
	# 2値化
	ret, thresh = cv2.threshold(im_gray, 127, 255, 0)
	# 角検出
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]
	
	for i in range(len(contours)) :
		if len(cnt) < len(contours[i]) :
			cnt = contours[i]
			
	
	# 凸包を描く
	hull = cv2.convexHull(cnt)
	cv2.drawContours(face_im, [hull], 0, (255, 255, 255), -2)

	return face_im

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


def poi(mask_path1, mask_path2, face_path1, face_path2):

	image = np.asarray(PIL.Image.open(mask_path1))
	image.flags.writeable = True
	image2 = np.asarray(PIL.Image.open(mask_path2))
	image2.flags.writeable = True
	
	image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
	
	# 両方共通のマスク
	res = cv2.bitwise_and(image, image2)
	
	face1 = np.asarray(PIL.Image.open(face_path1))
	face1.flags.writeable = True
	face2 = np.asarray(PIL.Image.open(face_path2))
	face2.flags.writeable = True

	face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
	
	ans = blend(face1, face2, res)
	ans = PIL.Image.fromarray(np.uint8(ans))
	
	return ans

# 実際に画像を処理していく (img_path : 読み込んだ画像のpath)
def cv_picture(img_path1, img_path2) :

	# 画像をpathから読み込む
	image1 = cv2.imread(img_path1)
	image2 = cv2.imread(img_path2)
	
	if image1.shape[0] > 500 and image1.shape[1] > 500 :
		image1 = cv2.resize(image1, (image1.shape[1] / 2, image1.shape[0] / 2))
		cv2.imwrite("./cgi-bin/img/a.jpg", image1)
		img_path1 = "./cgi-bin/img/a.jpg"
	if image2.shape[0] > 500 and image2.shape[1] > 500 :
		image2 = cv2.resize(image2, (image2.shape[1] / 2, image2.shape[0] / 2))
		cv2.imwrite("./cgi-bin/img/b.jpg", image2)
		img_path2 = "./cgi-bin/img/b.jpg"
	# グレースケール変換
	image_gray1 = cv2.cvtColor(image1, cv2.cv.CV_BGR2GRAY)
	image_gray2 = cv2.cvtColor(image2, cv2.cv.CV_BGR2GRAY)
	# 検出器の読み込み
	cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt_tree.xml")
	# 貌認識して、領域を保存(複数ある場合は配列になる)
	facerect1 = cascade.detectMultiScale(image_gray1, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
	facerect2 = cascade.detectMultiScale(image_gray2, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
	# 顔が複数あった場合の、画像保存名を分けるための変数
	global i
	i = 0
	
	# マスク画像を一時的に保存しておく場所
	IMAGE_UP = "./cgi-bin/image/"

	# 顔が一つ以上あったとき
	if len(facerect1) > 0 and len(facerect2) > 0 :
		global skin1, skin2
		global face1, face2
		
		# マスク作成
		skin_im, skin_mask = skin_detection(image1, facerect1, img_path1)
		skin_gray = cv2.cvtColor(skin_im, cv2.cv.CV_BGR2GRAY)

		
		# 顔をひとつひとつ読み込む(ループ)
		for rect in facerect1:
			# ファイルパスの更新
			face1 = "./cgi-bin/face/" + str(i) + os.path.basename(img_path1)
			#MASK_UP = "./cgi-bin/mask/" + str(i) + os.path.basename(Path)
			
			# 顔の画像情報
			img = image1[rect[1]:rect[1] + rect[2], rect[0]: rect[0] + rect[3]]
			img_gray = image_gray1[rect[1]:rect[1] + rect[2], rect[0]: rect[0] + rect[3]]

			cv2.imwrite(face1, img)
			i += 1
			
			s_im = skin_im[rect[1]:rect[1] + rect[2], rect[0]: rect[0] + rect[3]]
			s_g = skin_gray[rect[1]:rect[1] + rect[2], rect[0]: rect[0] + rect[3]]
			skin1 = cv_mask1(s_im, s_g, image1, rect)
			
			cv2.imwrite(IMAGE_UP + os.path.basename(img_path1), skin1)
			
		# マスク作成
		skin_im, skin_mask = skin_detection(image2, facerect2, img_path2)
		skin_gray = cv2.cvtColor(skin_im, cv2.cv.CV_BGR2GRAY)
			
		# 顔をひとつひとつ読み込む(ループ)
		for rect in facerect2 :
			# ファイルパスの更新
			face2 = "./cgi-bin/face/" + str(i) + os.path.basename(img_path2)
			#MASK_UP = "./cgi-bin/mask/" + str(i) + os.path.basename(Path)
			
			# 顔の画像情報
			img = image2[rect[1]:rect[1] + rect[2], rect[0]: rect[0] + rect[3]]
			img_gray = image_gray2[rect[1]:rect[1] + rect[2], rect[0]: rect[0] + rect[3]]

			cv2.imwrite(face2, img)
			i += 1
			
			s_im = skin_im[rect[1]:rect[1] + rect[2], rect[0]: rect[0] + rect[3]]
			s_g = skin_gray[rect[1]:rect[1] + rect[2], rect[0]: rect[0] + rect[3]]
			skin2 = cv_mask1(s_im, s_g, image1, rect)
			
			cv2.imwrite(IMAGE_UP + os.path.basename(img_path2), skin2)
		
		poisson_im = poi(IMAGE_UP + os.path.basename(img_path1), IMAGE_UP + os.path.basename(img_path2), face1, face2)
		path = "./cgi-bin/save/face.jpg"
		tmp = np.copy(facerect1[0])
		for rect in facerect1 :
			f = PIL.Image.open(img_path1)
			f.paste(poisson_im, (rect[0], rect[1]))
			if len(rect) >= len(tmp) :
				f.save(path)
			
		# 処理した画像の表示ページへ飛ぶボタン
		print '<form id="Form2" name="Form2" method="POST" action="./cgi-bin/pic.py">'
		print '<input type="text" value=', path, ' name="path"><br />'
		print '<input type="submit" value="編集画像を見る" name="submit">'
		print '</form>'
		print '<a href="../index.html">入力画面へ戻る</a>'

	else :
		print "<h1>顔を認識できませんでした</h1>"
		print "<br />"
		print '<a href="../index.html">入力画面へ戻る</a>'


if __name__ == "__main__" :
	
	# 特徴点検出 (SURF)
	detector = cv2.FeatureDetector_create("SURF")
	descriptor = cv2.DescriptorExtractor_create("SURF")
	matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")

	# index.htmlで送られてきた画像を保存するディレクトリ
	UPLOAD_DIR = "./cgi-bin/img/"
	
	# フォームから情報を変数formに保存 (cgi.FieldStorage())
	# formの構造は配列
	form = cgi.FieldStorage()
	# 画像情報を取得 (フォームのnameが添え字になる)
	fileitem1 = form["img1"]	#元画像
	fileitem2 = form["img2"]	#合成画像
	
	path1, path2 = save_upload_file(UPLOAD_DIR, fileitem1, fileitem2)
	print "<br /> 保存場所(元画像): ", path1, "<br /> 　　　　　 合成画像: ", path2, "<br /><br />"
	cv_picture(path1, path2)
	
	

print "</html></body>"