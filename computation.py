import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import PPoly, splder, splev, splrep


def compute_cob_angles(points, xb, xe, spline_degree = 5):
	"""_summary_

	Args:
		points (list[float]): list of central line points to be fitted into a curve
		xb (float): lower limit
		xe (flloat): uppert limit 
		spline_degree (int, optional): degree of the B-spline. Defaults to 5.
	"""
      
	print(f"Computing Cob angle ...")
	ys = [int(points['x'][i]) for i in points.index]
	# step_height = 5
	# xs = [i * step_height for i in range(len(ys))]
	xs = [int(points['y'][i]) for i in points.index]
	n = len(ys)

	smoothing = n - math.sqrt(2 * n)
	print(f"Number of data points: {n}")

	# ### defining B-spline representation fo central line points representing the spine curve
	# k = the degree of the spline; 
	# xb, xe - the interval to fit
	spine_curve = splrep(xs, ys, k=spline_degree, s=smoothing, xb=xb, xe=xe)

	# defining first derivative equation of the spline representation of the spine curve 
	spine_der = splder(spine_curve)

	# construct evenly spaced samples, calculated over the interval for displaying b-spline curve
	xx, xx_step = np.linspace(xs[0], xs[-1], xs[-1]-xs[0], retstep=True)
	print(f"xx_step: {xx_step}")

	yy = splev(xx, spine_curve)  # splev Evaluate a B-spline over xx for displaying
	yy1 = splev(xx, spine_der)  # splev Evaluate a B-spline over xx for displaying

	# ### finding minimums and maximums (possible appex) in spine values
	ppoly = PPoly.from_spline(spine_der)
	extremums_x = ppoly.roots(extrapolate=False)

	print(f"Number of extremums: {extremums_x.shape}\nextremums x: \n{extremums_x}")
	extremums_x = np.insert(extremums_x, 0, xx[0])
	extremums_x = np.insert(extremums_x, len(extremums_x), xx[-1])
	print(f"x extremums: {type(extremums_x)}\nextremums x: \n{extremums_x}")
	extremums_y = splev(extremums_x, spine_curve)

		
	len_extremums = len(extremums_x)

	max_angles = []
	min_x = min(xs)
	max_x = max(xs)
	lines_1 = []
	lines_2 = []
	max_angles_2 = []

	# for each extremum	looks at both sides and collects tangeent lines as tuple of [(x, y), (slope, coefficient)]
    # for each combination of tangents calculates the angel and find the maximum
	for k in range(1, len_extremums-1):
		slopes_1 = [((point_x, splev(point_x, spine_curve)),
					  tangent_line(curve_der=spine_der, point=(point_x,splev(point_x, spine_curve))) ) for point_x in np.linspace(extremums_x[k-1], extremums_x[k], num=len(range(int(extremums_x[k-1]), int(extremums_x[k]))) )]
		
		slopes_2 = [((point_x, splev(point_x, spine_curve)),
					  tangent_line(curve_der=spine_der, point=(point_x,splev(point_x, spine_curve))) ) for point_x in np.linspace(extremums_x[k], extremums_x[k+1], num=len(range(int(extremums_x[k]), int(extremums_x[k+1]))) )]
		angles = []
		for i in range(len(slopes_1)):
			for j in range(len(slopes_2)):
				(x_1, y_1), (slope_1, c_1) = slopes_1[i]
				(x_2, y_2), (slope_2, c_2) = slopes_2[j]
				angle_rad = compute_angle_from_slopes(slope_1, slope_2, True)
				angles.append(((x_1, y_1, slope_1, c_1), (x_2, y_2, slope_2, c_2), angle_rad))

		print(f"found {len(angles)} angles ")
		if len(angles) > 0:
			max_angle = max(angles, key=lambda x: x[-1])

		y_min_1 = point_eq(min_x, max_angle[0][2], max_angle[0][3])
		y_max_1 = point_eq(max_x, max_angle[0][2], max_angle[0][3])

		y_min_2 = point_eq(min_x, max_angle[1][2], max_angle[1][3])
		y_max_2 = point_eq(max_x, max_angle[1][2], max_angle[1][3])

		lines_1.append(([min_x, max_x], [y_min_1, y_max_1]))
		lines_2.append(([min_x, max_x], [y_min_2, y_max_2]))    
		max_angles.append((max_angle[-1], rad_to_deg(max_angle[-1]) ))
		
		epsilon = 100
		a_x = max_angle[0][0]
		a_y = max_angle[0][1]
		# line_1 = [a_x, a_y, a_x+epsilon, point_eq(min_x, max_angle[0][2], max_angle[0][3] )]
		line_1 = [a_x, a_y, min_x, point_eq(min_x, max_angle[0][2], max_angle[0][3] )]
		# line_1 = [min_x, y_min_1, max_x, y_max_1]
		# line_2 = [min_x, y_min_2, max_x, y_max_2]
		b_x = max_angle[1][0]
		b_y = max_angle[1][1]
		line_2 = [b_x, b_y, max_x, point_eq(max_x, max_angle[1][2], max_angle[1][3] )]
		max_angles_2.append( (rad_to_deg(max_angle[-1]), line_1, line_2, (extremums_x[k], extremums_y[k]) ) )

	for ma in max_angles:
		print(f"Max angle: {ma[0]} = {ma[1]} \N{DEGREE SIGN}C")     

	return(xx, yy, extremums_x, extremums_y, lines_1, lines_2, max_angles, max_angles_2)
	# fig, ax = plt.subplots()
	# plt.scatter(xs, ys, **{"color": "cyan", "marker": "."}, label="original")
	# plt.scatter(
	# 	extremums_x, extremums_y, **{"color": "orange", "marker": "o"}, label="Extremums"
	# )
	# plt.plot(xx, yy, **{"color": "blue", "ls": "-"}, label="B-spline")
	# # plt.plot(xx,yy1, **{'color': 'red', 'ls': '-.'}, label="B-spline deriv")

	# for i, line in enumerate(lines_1):
	# 	plt.plot(line[0], line[1], **{"color": "orange", "ls": "-"}, label=f"tangent_{i}")
	# for i, line in enumerate(lines_2):
	# 	plt.plot(line[0], line[1], **{"color": "red", "ls": "-"}, label=f"tangent_{i}")
	    
	# # plt.plot([min_x,max_x], [y_min_a,y_max_a], **{"color": "orange", "ls": "-"}, label="a")
	# # plt.plot([min_x,max_x], [y_min_b,y_max_b], **{"color": "pink", "ls": "-"}, label="b")

	# # for i, line in enumerate(tangents):
	# #     plt.plot(line[0], line[1], **{"color": "orange", "ls": "-"}, label=f"tangent_{i}")

	# plt.ylim([0, 130])  # range from 0 to crop_width
	# plt.xlim([0, 830])  # range from 0 to crop_height

	# plt.axis("equal")
	# plt.legend(loc="best", fancybox=True, shadow=True)
	# plt.show()




def tangent_line(curve_der, point: tuple[np.float64, np.float64]) -> tuple[np.ndarray, np.float64]:
    """Computes the slope and the coefficient of the tangent line at given point.

    Args:
        curve_der (spline): The derivative of the curve
        point (tuple(numpy.float64, numpy.float64)): given x and y of a point

    Returns:
        typlle[np.ndarray, np.float64]: tuple of the slope and the coefficient
	"""
    # line eq: y = slope * x + b
    # slope: curve' (derivative) at point x = point[0]
    # b = point[1]-slope*point[0]
    # print(f"point - {type(point)} - x - {type(point[0])} derivative - {type(curve_der)}")
    slope = splev(point[0], curve_der)
    # print(f"deriv at point ({point[0]}, {point[1]}) is {slope}")
    b = point[1] - slope * point[0]

    return (slope, b)


def point_eq(x: int, slope: np.ndarray, coef: np.float64) -> np.float64:
    """By the 'point-slope' formula calculates the y for a given x, slope and coefficient

    Args:
        x (int): x coordinate
        slope (np.ndarray): the slope ??? why it is np.ndarray
        coef (np.float64): the coefficient

    Returns:
        np.float64: y value from a line at x
	"""
    return slope*x + coef


def rad_to_deg(x: float) -> float:
    """Convert radians into degrees

    Args:
        x (float): angle in radians

    Returns:
        float: angle in degrees
    """
    return 180.0/math.pi * x


def compute_angle_from_slopes(ma: np.ndarray, mb: np.ndarray, in_rad: bool) -> float:
    """Computes the angle between two tangent lines given by there slopes;

    Args:
        ma (np.ndarray): The slope of line a
        mb (np.ndarray): The slope of line b
        in_rad (bool): if false returns the result in degrees; otherwise in radians;

    Returns:
        float: angle
    """
    angle_rad = math.atan(abs((ma-mb)/(1+ma*mb)))

    if not in_rad:
        return rad_to_deg(angle_rad)
    
    return angle_rad


def find_central_line(spine_crop):
	image_h, image_w = spine_crop.shape
	window_w = 60
	# window_w = int(image_w*0.33)
	window_h = 10
	step_w = 1
	step_h = 5
	c_y = c_x = 0
	central_line_points = []
	i = 0
	while c_y < image_h - window_h:
		max_sum = max_sum = (0, 0, 0)
		c_x = 0
		while c_x < image_w - window_w:
			roi = spine_crop[c_y:c_y+window_h, c_x:c_x+window_w]
			current_sum  = np.sum(roi)
			if max_sum[0] < current_sum:
				curr_x = c_x+window_w//2
				
				max_sum = (current_sum, curr_x, c_y)
			c_x += step_w
		central_line_points.append((max_sum[1],max_sum[2]))
		c_y += step_h
	return central_line_points


def refine_central_line(central_line_points, epsilon):
	central_line_points_processed = []
	# prev_x = image_width//2
	prev_x = central_line_points[0][0]
	central_line_points_processed.append((prev_x, central_line_points[0][1]))
	i = 1
	for point in central_line_points[1:]:
		curr_x = point[0]
		curr_y = point[1]
		prev_point = central_line_points[i-1]

		diff = curr_x - prev_x
		if abs(diff) > epsilon:
			# curr_x = curr_x + diff//2
			curr_x = prev_x
		point = (curr_x, point[1])
		central_line_points_processed.append(point)
		i += 1
		prev_x = curr_x

	return central_line_points_processed

@dataclass
class Point:
	x: float
	y: float

@dataclass
class Line:
	a: Point
	b: Point

@dataclass
class CobAngle:
	line_a: Line
	line_b: Line
	apex: Point
	measure: float