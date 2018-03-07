import numpy as np
import cv2
import annotation as atn
import matplotlib.pyplot as plt
import math
import sys
import random



def intercept(a):
	if(float(a[2])-float(a[0])==0):
		return 9999999
	slope = (float(a[3])-float(a[1]))/(float(a[2])-float(a[0]))
	return -slope*float(a[0]) + float(a[1])

def line_detect(image,threshold = 100):
	# image = cv2.imread(path)
	# img = cv2.resize(image,(800,600))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gaus = cv2.GaussianBlur(gray,(5,5),0) 
	edges = cv2.Canny(gaus, 50, 150, apertureSize=3)
	minLineLength = 300  
	maxLineGap = 0.8
	# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength, maxLineGap)  
	lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold,srn = 100, stn = 0.1)
	lines = np.squeeze(lines,axis=1)
	lines = np.array([x for x in lines if (x[1]>1.4 and x[1]<1.7)])
	print type(lines)
	return lines


# def high_frequency(input_array, k):
# 	a = [0] * k
# 	for i in input_array:	
# 		a[i[0]] += 1
# 	return np.argmax(a)

def line_distance(line_dot):
	return math.sqrt((line_dot[0]-line_dot[2])**2 + (line_dot[1]-line_dot[3])**2)

def line_slope(line_dot):

	if line_dot[0] == line_dot[2]:
		return sys.maxint
	else:
		return (line_dot[1] - line_dot[3]) / (line_dot[0] - line_dot[2])


def line_detect_Hough_p(image,annotation,threshold = 12):
	lines_dot = []
	img = cv2.resize(annotation,(1280,960))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gaus = cv2.GaussianBlur(gray,(5,5),2.5) 
	edges = cv2.Canny(gaus, 50, 150, apertureSize=3)
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength = 150, maxLineGap = 50)  
	lines = np.squeeze(lines,axis=1)
	slope_lines = np.float32(lines)
	lines = []
	distance = []
	
	for i in slope_lines:
		if  (annotation[int(i[1])][int(i[0])] == 255 and annotation[int(i[3])][int(i[2])] == 255\
			and annotation[int((i[1]+i[3])/2)][int((i[0]+i[2])/2)] == 255):
			distance.append(line_distance(i))
			lines.append([int(i[0]),int(i[1]),int(i[2]),int(i[3])])
	
	lines = np.array(lines)
	longest_index = np.argmax(distance)
	result = []
	for i in lines:	
		if abs( line_slope(i)-line_slope( lines[longest_index]) ) < 0.1:
			# cv2.line(image,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(0,0,255),2) 
			# cv2.line(annotation,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),255,2)
			result.append([int(i[0]),int(i[1]),int(i[2]),int(i[3])])
	return result


def dis_dot_to_line(line, dot):
	line = np.float32(line)
	dot = np.float32(dot)
	slope = (line[1]-line[3])/(line[0]-line[2])
	return abs(slope*dot[0]-dot[1]-slope*line[2]+line[3]) / math.sqrt(slope**2+1) + \
			abs(slope*dot[2]-dot[3]-slope*line[2]+line[3]) / math.sqrt(slope**2+1)

def kmeans(dot, iter_num = 100):
	min_loss = sys.maxint
	
	for iteration in range(iter_num):
		
		dot_copy = [i for i in dot]
		result_dot = [[],[],[]]
		distance_sum = [0,0,0]
		
		for i in range(3):
			standard = random.choice(dot_copy)
			result_dot[i].append(standard)
			dot_copy.remove(standard)
		
		
		for i in dot_copy:
			distance = [dis_dot_to_line(result_dot[j][0], i) for j in range(3)]
			index = np.argmin(distance)
			result_dot[index].append(i)
			distance_sum[index] +=  distance[index]
		
		distance_mean = [ distance_sum[i]/float(len(result_dot[i])) for i in range(3)] 
		
		if sum(distance_mean) < min_loss:
			min_loss = sum(distance_mean)
			# print min_loss
			result = [ i for i in result_dot]
	
	dot_array = np.array(dot)
	plt.scatter(dot_array[:,0],-dot_array[:,1], c='c',marker = '*', linewidths = 8)
	plt.scatter(dot_array[:,2],-dot_array[:,3], c='c',marker = '*', linewidths = 8)	
	return result

def plot_line_kmeans(img, dot_list):
	for i in range(len(dot_list)):
		dot = []
		for j in dot_list[i]:
			dot.append([j[0],j[1]])
			dot.append([j[2],j[3]])
		[vx,vy,x,y] = cv2.fitLine(np.array(dot), cv2.DIST_L2, 0, 0.01, 0.01)
		lefty = int((-x*vy/vx) + y)
		righty = int(((img.shape[1]-x)*vy/vx) + y)
		cv2.line(img, (img.shape[1]-1,righty), (0,lefty),\
			(255*(i==0), 255*(i==1), 255*(i==2)), 2)
	return 0 



def scatter_1(dot_list):
	f1 = plt.figure(1) 
	for i in range(len(dot_list)):
		for j in (dot_list[i]):
			if i==0:
				plt.scatter(j[0], -j[1], c="b")
				plt.scatter(j[2], -j[3], c="b")
			if i==1:
				plt.scatter(j[0], -j[1], c="g")
				plt.scatter(j[2], -j[3], c="g")
			if i==2:
				plt.scatter(j[0], -j[1], c="r")	
				plt.scatter(j[2], -j[3], c="r")			
	plt.xlim(0,1280)	
	plt.ylim(-960,0)	
	plt.show()
	return 0


def dilate(img):
	image = cv2.resize(img,(1280,960))
	kernel=np.uint8(np.zeros((11,11)))  
	for x in range(11):  
	    kernel[x,5]=1;  
	    kernel[5,x]=1;  
	dilated = cv2.dilate(image,kernel) 
	for i in range(1):
		dilated = cv2.dilate(dilated,kernel)
	return dilated 






# img_path = "Data_zoo/MIT_SceneParsing/ADEChallengeData2016/picture/" + str(num) + ".jpg"
# annotation_path = "Data_zoo/MIT_SceneParsing/ADEChallengeData2016/annotations/training/" + str(num) +".png"
# img_path = "test_picture/4.jpg"
annotation_path = "pb/prediction.png"

def test():
	for i in range(20):
		num = i+1
		img_path = "test_picture/"+str(num)+".jpg"
		image = cv2.imread(img_path)
		resize_image = cv2.resize(image,(1280,960))
		atn.FCN(img_path)
		annotation = cv2.imread(annotation_path,cv2.IMREAD_GRAYSCALE)
		dilated_img = dilate(annotation)
		ret, binary_img = cv2.threshold(dilated_img, 50, 255, cv2.THRESH_BINARY) 
		line = line_detect_Hough_p(resize_image,binary_img)
		classfication_line = kmeans(line,100)
		plot_line_kmeans(resize_image, classfication_line)

		# scatter_1(classfication_line)
		# cv2.imshow("dilated_img",dilated_img)
		# cv2.imshow("binary_img",binary_img)
		cv2.imshow("houghline",resize_image)
		cv2.imwrite("test_result/"+str(num)+".jpg", resize_image)
		cv2.waitKey(1)	
	cv2.destroyAllWindows() 	
test()


