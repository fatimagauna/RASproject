import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Select a region of interest
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon formed
    by the vertices. The rest of the image pixels is set to zero (black).
    """

    # Defining a blank mask
    mask = np.zeros_like(img)

    # Define a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Return the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def pipeline(id_cam):

	vid = cv2.VideoCapture(id_cam)
			       
	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, img_colour = cap.read()

		if ret == True:
			#Verificar imagen
			#img_colour=cv2.imread(img_name)

			if img_colour is None:
				print ('ERROR IMAGEN', img_name,'could not read')
				exit()

			# Converti imagen a gris
			img_colour_rgb=cv2.cvtColor(img_colour,cv2.COLOR_BGR2RGB)
			grey = cv2.cvtColor(img_colour_rgb,cv2.COLOR_RGB2GRAY)
			cv2.imshow("Lane line detection",grey)


			#MOdificar tamano
			scale_percent=30
			width =int(grey.shape[1]* scale_percent/100)
			height=int(grey.shape[1]* scale_percent/100)
			dim=(width,height)
			grey=cv2.resize(grey,dim,interpolation=cv2.INTER_AREA)
			cv2.imshow("Lane line detection",grey)


			# 3.- Apply Gaussian smoothing
			kernel_size = (17, 17)
			blur_grey = cv2.GaussianBlur(grey, kernel_size, sigmaX=0, sigmaY=0)
			cv2.imshow("Smoothed image", blur_grey)


			#EDge detector
			low_threshold = 70
			high_threshold = 100
			edges = cv2.Canny(blur_grey, low_threshold, high_threshold, apertureSize=3)
			cv2.imshow("Canny image", edges)


			#EScoger vertices para delimitar areas
			p1 = (6, 527)
			p2 = (3, 396)
			p3 = (463, 369)
			p4 = (695, 375)
			p5 = (1091, 417)
			p6 = (1098, 537)

			# Create a vertices array that will be used for the roi
			vertices = np.array([[p1, p2, p3, p4, p5, p6]], dtype=np.int32)


			masked_edges = region_of_interest(edges, vertices)
			cv2.imshow("Canny Image with region of interes", masked_edges)


			# 7.- Apply Hough transform for lane lines detection
			rho = 0.5                     # distance resolution in pixels of the Hough grid
			theta = np.pi/1000            # angular resolution in radians of the Hough grid
			threshold = 80                # minimum number of votes (intersections in Hough grid cell)
			min_line_len = 2              # minimum number of pixels making up a line
			max_line_gap = 3              # maximum gap in pixels between connectable line segments
			line_image = np.copy(img_colour)*0   # creating a blank to draw lines on
			hough_lines = cv2.HoughLinesP(masked_edges,
						      rho,
						      theta,
						      threshold,
						      np.array([]),
						      minLineLength=min_line_len,
						      maxLineGap=max_line_gap)

			print("number of lines:{}".format(hough_lines.shape))


			# 8.- Initialise a new image to hold the original image with the detected lines
			# Resize img_colour_with_lines
			scale_percent = 30
			width = int(img_colour_rgb.shape[1] * scale_percent / 100)
			height = int(img_colour_rgb.shape[0] * scale_percent / 100)
			dim = (width, height)
			img_colour_rgb = cv2.resize(img_colour_rgb, dim, interpolation = cv2.INTER_AREA)
			img_colour_with_lines = img_colour_rgb.copy()

			left_lines, left_slope, right_lines, right_slope = list(), list(), list(), list()
			ymin, ymax, xmin, xmax = 0.0, 0.0, 0.0, 0.0
			x_left, y_left, x_right, y_right = list(), list(), list(), list()

			# Slope and standard deviation for left and right lane lines
			left_slope_mean, left_slope_std = -20.09187457328413, 3.4015553620470467
			right_slope_mean, right_slope_std = 21.713840954352456, 1.7311898404656396

			# Loop through each detected line
			for line in hough_lines:
                    		for x1, y1, x2, y2 in line:
					slope = (y2 - y1) / (x2 - x1) #Pendiente
					if math.fabs(slope) < param['extremeSlope']: # <-- Only consider extreme slope
					    continue   #Continue para ignorar esta pendiente y seguir con la siguiente
					if slope <= 0: # <-- Si es negativo es linea del lado izquierda
					    left_line_x.extend([x1, x2])
					    left_line_y.extend([y1, y2])
					else: # <-- Si no es de la derecha
					    right_line_x.extend([x1, x2])
					    right_line_y.extend([y1, y2])

					# Otherwise, the current line belongs to the left lane line
					elif (slope_deg >= (left_slope_mean - 1*left_slope_std)) and (slope_deg < (left_slope_mean + 1*left_slope_std)):
					    left_lines.append(line)
					    cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (0,0,255), 10)
					    left_slope.append(slope)
					    x_left.append(x1)
					    x_left.append(x2)
					    y_left.append(y1)
					    y_left.append(y2)

					# Outliers lines; i.e., lines that neither belong to left nor right lane lines
					else:
					    pass

			cv2.imshow("Canny image with detected lines", img_colour_with_lines)

			x_min = np.min(x_left)
			x_max = np.max(x_left)
			y_min = np.min(y_left)
			y_max = np.max(y_left)

			# Find the regression line for the left lane line
			left_regression_line = linear_model.LinearRegression()
			left_regression_line.fit(np.array(x_left).reshape(-1,1), y_left)

			y_bottom = left_regression_line.coef_*x_min + left_regression_line.intercept_
			y_top = left_regression_line.coef_*x_max + left_regression_line.intercept_

			# Draw a green line to depict the left lane line
			cv2.line(img_colour_with_lines, (x_min, int(y_bottom[0])), (x_max, int(y_top[0])), (0,255,0), 5)

			x_min = np.min(x_right)
			x_max = np.max(x_right)
			y_min = np.min(y_right)
			y_max = np.max(y_right)

			# Fidn the regression line for the right lane line
			right_regression_line = linear_model.LinearRegression()
			right_regression_line.fit(np.array(x_right).reshape(-1,1), y_right)

			y_bottom = right_regression_line.coef_*x_min + right_regression_line.intercept_
			y_top = right_regression_line.coef_*x_max + right_regression_line.intercept_
			# Draw a green line to depict the right lane line
			cv2.line(img_colour_with_lines, (x_min, int(y_bottom[0])), (x_max, int(y_top[0])), (0,255,0), 5)

			cv2.imshow("Canny image with detected lines", img_colour_with_lines)

			cv2.waitKey(0)

id_cam=0

pipeline(id_cam)
