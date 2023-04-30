import random
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time



def euclidean_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_point_inside_circle(circle, point, tol=1e-9):
    center, radius = circle
    dx, dy = point[0] - center[0], point[1] - center[1]
    distance = np.sqrt(dx*dx + dy*dy)
    return distance <= radius + tol

def circle_from_two_points(p1, p2):
    center = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    radius = euclidean_distance(center, p1)
    return (center, radius)


def circle_from_three_points(p1, p2, p3):
    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

    # Check if the points are collinear
    if d == 0:
        return (0, 0), float('inf')

    ux = ((p1[0] ** 2 + p1[1] ** 2) * (p2[1] - p3[1]) + (p2[0] ** 2 + p2[1] ** 2) * (p3[1] - p1[1]) + (p3[0] ** 2 + p3[1] ** 2) * (p1[1] - p2[1])) / d
    uy = ((p1[0] ** 2 + p1[1] ** 2) * (p3[0] - p2[0]) + (p2[0] ** 2 + p2[1] ** 2) * (p1[0] - p3[0]) + (p3[0] ** 2 + p3[1] ** 2) * (p2[0] - p1[0])) / d
    center = np.array([ux, uy])

    radius = np.linalg.norm(center - p1)

    return center, radius


def min_circle(points, support_points, step_circles=None):
    if step_circles is None:
        step_circles = []

    # Base cases: if there are no points left to process or we have 3 support points
    if not points or len(support_points) == 3:
        circle = circle_from_support_points(support_points)
        # Add the circle to the list of step_circles
        step_circles.append(circle)
        return circle

    # Remove the last point from the list of points
    p = points.pop()
    # Recursively call min_circle without the point p
    circle = min_circle(points, support_points, step_circles)
    if not is_point_inside_circle(circle, p):
        # If p is outside the circle, add it to the support points
        support_points.append(p)
        # Recursively call min_circle with the updated support_points
        circle = min_circle(points, support_points, step_circles)
        # Remove the point p from the support points
        support_points.pop()
    points.append(p)
    return circle



def circle_from_support_points(points):
    if not points:
        return (0, 0), 0  # return a default circle if no points are given
    if len(points) == 1:
        return points[0], 0
    if len(points) == 2:
        return circle_from_two_points(points[0], points[1])
    if len(points) == 3:
        return circle_from_three_points(points[0], points[1], points[2])

    # For any other case, return a default circle
    return (0, 0), 0

def welzl(points, step_circles=None):
    random.shuffle(points)
    # Initialize an empty list to store intermediate circles
    circles = []

    # Iterate through the points from index 1 to the length of the points list
    for i in range(1, len(points) + 1):
        # Call the min_circle function for each subset of points from the beginning up to index i
        # with an empty list of support points
        circles.append(min_circle(points[:i], []))

    # If step_circles is provided, extend it with the intermediate circles
    if step_circles is not None:
        step_circles.extend(circles)

    # Return the last circle in the circles list, which is the minimum enclosing circle
    return circles[-1]

def skyum(points):
    def polar_angle(p0, p1):
        return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

    # Base case: if there are less than 2 points, create a circle from the support points
    if len(points) < 2:
        return circle_from_support_points(points)

    leftmost_point = min(points, key=lambda p: p[0])
    # Sort the points by their polar angle with respect to the leftmost point
    sorted_points = sorted(points, key=lambda p: polar_angle(leftmost_point, p))

    # Create an initial circle from the first two points in the sorted list
    circle = circle_from_two_points(sorted_points[0], sorted_points[1])

    # Iterate through the sorted points starting from index 2
    for i in range(2, len(sorted_points)):
        # Check if the current point is inside the circle
        if not is_point_inside_circle(circle, sorted_points[i]):
            # If not, iterate through the points again up to index i
            for j in range(i):
                # Create a new circle from the two points at indices j and i
                circle_2 = circle_from_two_points(sorted_points[j], sorted_points[i])
                # Check if all points up to index i are inside circle_2
                if all(is_point_inside_circle(circle_2, p) for p in sorted_points[:i]):
                    # If yes, update the circle to be circle_2
                    circle = circle_2
                else:
                    # If not, iterate through the points again up to index j
                    for k in range(j):
                        # Create a new circle from the three points at indices k, j, and i
                        circle_3 = circle_from_three_points(sorted_points[k], sorted_points[j], sorted_points[i])
                        # Check if all points up to index i are inside circle_3
                        if all(is_point_inside_circle(circle_3, p) for p in sorted_points[:i]):
                            # If yes, update the circle to be circle_3 and break the loop
                            circle = circle_3
                            break
    return circle


def brute_force_min_circle(points):
    min_circle = None
    min_radius = float('inf')

    for p1 in points:
        for p2 in points:
            if p1 != p2:
                circle = circle_from_two_points(p1, p2)
                if all(is_point_inside_circle(circle, p) for p in points):
                    if circle[1] < min_radius:
                        min_radius = circle[1]
                        min_circle = circle

                for p3 in points:
                    if p3 != p1 and p3 != p2:
                        circle = circle_from_three_points(p1, p2, p3)
                        if all(is_point_inside_circle(circle, p) for p in points):
                            if circle[1] < min_radius:
                                min_radius = circle[1]
                                min_circle = circle

    return min_circle

def generate_random_points(num_points, x_range=(-10, 10), y_range=(-10, 10), distribution='uniform'):
    x_min, x_max = x_range
    y_min, y_max = y_range
    if distribution == 'uniform':
        random_points = np.random.uniform((x_min, y_min), (x_max, y_max), size=(num_points, 2))
    elif distribution == 'normal':
        x_mean, x_std = (x_max + x_min) / 2, (x_max - x_min) / 4
        y_mean, y_std = (y_max + y_min) / 2, (y_max - y_min) / 4
        random_points = np.random.normal(loc=[x_mean, y_mean], scale=[x_std, y_std], size=(num_points, 2))
    elif distribution == 'exponential':
        x_scale, y_scale = (x_max - x_min) / 2, (y_max - y_min) / 2
        random_points = np.random.exponential((x_scale, y_scale), size=(num_points, 2))
    elif distribution == 'poisson':
        x_lam, y_lam = (x_max - x_min) / 2, (y_max - y_min) / 2
        random_points = np.random.poisson((x_lam, y_lam), size=(num_points, 2))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return [tuple(point) for point in random_points]


def run_tests(point_counts, num_trials):
    distributions = ["uniform", "normal", "exponential", "poisson"]
    all_welzl_times = {}
    all_skyum_times = {}

    for distribution in distributions:
        welzl_times = []
        skyum_times = []

        for count in point_counts:
            welzl_results = []
            skyum_results = []

            for _ in range(num_trials):
                points = generate_random_points(count, distribution=distribution)

                start_time = time.time()
                _, _ = welzl(points)
                welzl_time = time.time() - start_time
                welzl_results.append(welzl_time)

                start_time = time.time()
                _, _ = skyum(points)
                skyum_time = time.time() - start_time
                skyum_results.append(skyum_time)

            welzl_times.append(welzl_results)
            skyum_times.append(skyum_results)

        all_welzl_times[distribution] = welzl_times
        all_skyum_times[distribution] = skyum_times

    return all_welzl_times, all_skyum_times


def plot_candlestick_chart(point_counts, all_welzl_times, all_skyum_times):
    width_per_distribution = 0.6
    positions = np.arange(len(point_counts)) * 3

    for distribution in all_welzl_times:
        welzl_times = all_welzl_times[distribution]
        skyum_times = all_skyum_times[distribution]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f'Comparison of Welzl\'s Algorithm and Skyum Algorithm for {distribution.capitalize()} Distribution')
        ax.set_xlabel('Number of Points')
        ax.set_ylabel('Execution Time (seconds)')

        welzl_box = ax.boxplot(welzl_times, positions=positions - 0.6, widths=width_per_distribution, patch_artist=True, boxprops=dict(facecolor="C0"))
        skyum_box = ax.boxplot(skyum_times, positions=positions + 0.6, widths=width_per_distribution, patch_artist=True, boxprops=dict(facecolor="C1"))

        ax.legend([welzl_box['boxes'][0], skyum_box['boxes'][0]], ['Welzl\'s Algorithm', 'Skyum Algorithm'])
        ax.set_xticks(positions)
        ax.set_xticklabels(point_counts)
        plt.grid()

        plt.savefig(f"boxplot_welzl_vs_skyum_{distribution}.png")


point_counts = [5, 10, 20, 50, 100, 200] #, 300, 400, 500]
num_trials = 3

welzl_times, skyum_times = run_tests(point_counts, num_trials)
plot_candlestick_chart(point_counts, welzl_times, skyum_times)
