import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

def Velocidad(xmil , xmal, y_bl, y_tl, xmir, xmar, y_br, y_tr):
    #Left Lane
    xminl=xmil
    xmaxl=xmal
    y_bottoml=y_bl
    y_topl=y_tl
    #Right Lane
    xminr=xmir
    xmaxr=xmar
    y_bottomr=y_br
    y_topr=y_tr
    #Pendiente
    ml=(-y_topl+y_bottoml)/(xmaxl-xminl)
    mr=(-y_topr+y_bottomr)/(xmaxr-xminr)
    #B's
    bl=y_topl-ml*xminl
    br=y_bottomr-mr*xmaxr
    #Pendientes impresas
    print("\nLas fórmulas de las rectas izq y der son: ")
    print("y=",ml,"x +",bl)
    print("y=",mr,"x +",br,"\n")
    #Calculo de xr y xl en ya
    ya=250 
    xl=(ya-bl)/ml
    xr=(ya-br)/mr
    #Xroi
    xroi=(xr-xl)/2+xl
    print("xroi es: ",xroi)
    A = np.matrix([[-ml, 1],[-mr, 1]])
    b = np.matrix([[bl],[br]])
    [yint,xint] = (A**-1)*b
    print("Las coordenadas son: ",xint)


    constRap=0.5
    if (xroi < xint):
        correcion = (abs(xroi-xint)*constRap)/xroi
        rapIzq=constRap-correcion
        rapDer=constRap+correcion
        print(rapIzq, rapDer)
    elif (xroi > xint):
        correcion = ((xroi-xint)*constRap)/xroi
        rapIzq=constRap+correcion
        rapDer=constRap-correcion
        print(rapIzq, rapDer)
    else:
        rapIzq=constRap
        rapDer=constRap
       
    return [rapIzq, rapDer]
    
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
    

    
cap = cv2.VideoCapture(0, cv2.CAP_V4L)


while(cap.isOpened()):

    ret, frame = cap.read()
    if (ret==True):
    #frame = cv2.resize(frame, (424,240))

        #2. COnvertir de BGR a RGB, luego de RGB a Gray
        img_colour_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        grey = cv2.cvtColor(img_colour_rgb, cv2.COLOR_RGB2GRAY)

        #Resize image
        scale_percent = 20
        width = int(grey.shape[1]* scale_percent / 100)
        height = int(grey.shape[0]* scale_percent / 100)
        dim = (width, height)
        grey = cv2.resize(grey, dim, interpolation = cv2.INTER_AREA)
        #cv2.imshow("Lane line detection", grey)

        # 3.- Apply Gaussian smoothing
        kernel_size = (17,17)
        blur_grey = cv2.GaussianBlur(grey, kernel_size, sigmaX=0, sigmaY=0)
        #cv2.imshow("Smoothed image", blur_grey)

        # 4.- Apply Canny edge detector
        low_threshold = 70
        high_threshold = 100
        edges = cv2.Canny(blur_grey, low_threshold, high_threshold, apertureSize=3)
        #cv2.imshow("Canny image", edges)


        # 5.- Define a polygon-shape like region of interest
        img_shape = grey.shape

        # RoI (change the below vertices accordingly)
        p1 = (1, 340)
        p2 = (3, 249)
        p3 = (317, 184)
        p4 = (440, 185)
        p5 = (723, 281)
        p6 = (728, 342)

        # Create a vertices array that will be used for the roi
        vertices = np.array([[p1, p2, p3, p4, p5, p6]], dtype=np.int32)

        #6. Get region of interest using the just created polygon.
        # This will be used together with the Hugh 

        masked_edges = region_of_interest(edges, vertices)
        #cv2.imshow("Canny image within Region Of Interest", masked_edges)

        # 7.- Apply Hough transform for lane lines detection
        rho = 0.5                     # distance resolution in pixels of the Hough grid
        theta = np.pi/1000            # angular resolution in radians of the Hough grid
        threshold = 80                # minimum number of votes (intersections in Hough grid cell)
        min_line_len = 2              # minimum number of pixels making up a line
        max_line_gap = 3              # maximum gap in pixels between connectable line segments
        line_image = np.copy(img_colour_rgb)*0   # creating a blank to draw lines on
        hough_lines = cv2.HoughLinesP(masked_edges,
                                      rho,
					                  theta,
					                  threshold,
					                  np.array([]),
					                  minLineLength=min_line_len,
					                  maxLineGap=max_line_gap)               


        # 8.- Initialise a new image to hold the original image with the detected lines
        # Resize img_colour_with_lines
        scale_percent = 20
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
        
        if hough_lines == None:
            print("No hay nada")
            quit()
        # Loop through each detected line
        for line in hough_lines:
            for x1, y1, x2, y2 in line:

                # Compute slope for current line
                slope = (y2-y1) / (x2-x1)
                #print(slope)
                slope_deg = np.rad2deg(np.arctan(slope))
                #cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (0,0,255), 10)

                # If slope is positive, the current line belongs to the right lane line
                if (slope_deg >= (right_slope_mean - 1*right_slope_std)) and (slope_deg < (right_slope_mean + 1*right_slope_std)):
                    right_lines.append(line)
                    #cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (255,0,0), 10)
                    right_slope.append(slope)

                    x_right.append(x1)
                    x_right.append(x2)
                    y_right.append(y1)
                    y_right.append(y2)

                # Otherwise, the current line belongs to the left lane line
                elif (slope_deg >= (left_slope_mean - 1*left_slope_std)) and (slope_deg < (left_slope_mean + 1*left_slope_std)):
                    left_lines.append(line)
                    #cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (0,0,255), 10)
                    left_slope.append(slope)
                    x_left.append(x1)
                    x_left.append(x2)
                    y_left.append(y1)
                    y_left.append(y2)

                # Outliers lines; i.e., lines that neither belong to left nor right lane lines
                else:
                    pass

        cv2.imshow("Canny image with detected lines", img_colour_with_lines)

        x_minl = np.min(x_left)
        x_maxl = np.max(x_left)
        y_minl = np.min(y_left)
        y_maxl = np.max(y_left)

        # Find the regression line for the left lane line
        left_regression_line = linear_model.LinearRegression()
        left_regression_line.fit(np.array(x_left).reshape(-1,1), y_left)

        y_bottoml = left_regression_line.coef_*x_minl + left_regression_line.intercept_
        y_topl = left_regression_line.coef_*x_maxl + left_regression_line.intercept_
        print("Left Line")
        print("xmin: ",x_minl)
        print("xmax: ",x_maxl)
        print("y bottom: ",y_bottoml[0])
        print("y top: ",y_topl[0])

        # Draw a green line to depict the left lane line
        cv2.line(img_colour_with_lines, (x_minl, int(y_bottoml[0])), (x_maxl, int(y_topl[0])), (0,255,0), 5)

        x_minr = np.min(x_right)
        x_maxr = np.max(x_right)
        y_minr = np.min(y_right)
        y_maxr = np.max(y_right)

        # Fidn the regression line for the right lane line
        right_regression_line = linear_model.LinearRegression()
        right_regression_line.fit(np.array(x_right).reshape(-1,1), y_right)

        y_bottomr = right_regression_line.coef_*x_minr + right_regression_line.intercept_
        y_topr = right_regression_line.coef_*x_maxr + right_regression_line.intercept_
        # Draw a green line to depict the right lane line
        cv2.line(img_colour_with_lines, (x_minr, int(y_bottomr[0])), (x_maxr, int(y_topr[0])), (0,255,0), 5)

        cv2.imshow("Canny image with detected lines", img_colour_with_lines)
        print("Right Line")
        print("xmin: ",x_minr)
        print("xmax: ",x_maxr)
        print("y bottom: ",y_bottomr[0])
        print("y top: ",y_topr[0])
        
        Velocidad(x_minl, x_maxl, y_bottoml[0], y_topl[0], x_minr, x_maxr, y_bottomr[0], y_topr[0])

        cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
