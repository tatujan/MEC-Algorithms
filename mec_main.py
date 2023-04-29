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

    if not points or len(support_points) == 3:
        circle = circle_from_support_points(support_points)
        step_circles.append(circle)
        return circle

    p = points.pop()
    circle = min_circle(points, support_points, step_circles)
    if not is_point_inside_circle(circle, p):
        support_points.append(p)
        circle = min_circle(points, support_points, step_circles)
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

def min_disk_with_two_points(points, q1, q2):
    circle = circle_from_two_points(q1, q2)
    inside_points = [p for p in points if is_point_inside_circle(circle, p)]
    if len(inside_points) == len(points):
        return circle
    return None

def welzl(points, step_circles=None):
    random.shuffle(points)
    circles = []
    for i in range(1, len(points) + 1):
        circles.append(min_circle(points[:i], []))
    if step_circles is not None:
        step_circles.extend(circles)
    return circles[-1]

def skyum(points):
    def polar_angle(p0, p1):
        return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

    if len(points) < 2:
        return circle_from_support_points(points)

    leftmost_point = min(points, key=lambda p: p[0])
    sorted_points = sorted(points, key=lambda p: polar_angle(leftmost_point, p))

    circle = circle_from_two_points(sorted_points[0], sorted_points[1])
    for i in range(2, len(sorted_points)):
        if not is_point_inside_circle(circle, sorted_points[i]):
            for j in range(i):
                circle_2 = circle_from_two_points(sorted_points[j], sorted_points[i])
                if all(is_point_inside_circle(circle_2, p) for p in sorted_points[:i]):
                    circle = circle_2
                else:
                    for k in range(j):
                        circle_3 = circle_from_three_points(sorted_points[k], sorted_points[j], sorted_points[i])
                        if all(is_point_inside_circle(circle_3, p) for p in sorted_points[:i]):
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


# Test
# num_points = 100
# points = generate_random_points(num_points, x_range=(-10, 10), y_range=(-10, 10))
points = [(1, 2), (2, 2), (5, 1), (3, 2), (4, 4), (4, 1)]

# Visualization
def plot_circle(circle, ax):
    center, radius = circle
    if center is None:
        return
    circle_patch = plt.Circle(center, radius, fill=False, linewidth=2)
    ax.add_patch(circle_patch)


def visualize(points, welzl_circle, skyum_circle):
    fig, ax = plt.subplots()
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_aspect("equal")

    # Plot the points
    px, py = zip(*points)
    ax.scatter(px, py, color="red")

    # Plot the minimum enclosing circles
    plot_circle(welzl_circle, ax, linestyle="-", edgecolor="blue", facecolor="none")
    # plot_circle(skyum_circle, ax, linestyle="--", edgecolor="green", facecolor="none")

    # ax.legend(["Welzl's Algorithm", "Skyum's Algorithm"], loc="upper left")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Minimum Enclosing Circle - Welzl's and Skyum's Algorithms")

    plt.show()

def visualize_steps(points, step_circles, save_as_gif=True):
    fig, ax = plt.subplots()

    ax.set_aspect("equal")

    def update(num):
        ax.clear()
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        # ax.set_xlim((min(points, key=lambda p: p[0])[0])*2-2, (max(points, key=lambda p: p[0])[0])*2+2)
        # ax.set_ylim((min(points, key=lambda p: p[1])[1])*2-2, (max(points, key=lambda p: p[1])[1])*2+2)
        ax.scatter(*zip(*points[:num+1]), color="red")
        plot_circle(step_circles[num], ax)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(f"Minimum Enclosing Circle - Welzl's Algorithm - Step {num+1}")

    ani = FuncAnimation(fig, update, frames=range(len(step_circles)), repeat=False, interval=1000)

    if save_as_gif:
        ani.save('animation_welzl.gif', writer='pillow', fps=1)
    else:
        ani.save('animation.mp4', writer='ffmpeg', fps=1)


    plt.close(fig)


step_circles = []
center, radius = welzl(points, step_circles=step_circles)
visualize_steps(points, step_circles, save_as_gif=True)
