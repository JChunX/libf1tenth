import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import CubicSpline

def compute_theta(x, y):
    theta = []
    for i in range(len(x)):
            theta.append(math.atan2(y[i] - y[i - 1], x[i] - x[i - 1]))
    return theta

def compute_derivative(x, y):
    ss = [0.0]
    dists = []
    cum_dist = 0.0
    for i in range(1, len(x)):
        dist = math.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
        dists.append(dist)
        cum_dist += dist
        ss.append(cum_dist)

    dists.append(math.sqrt((x[0] - x[-1]) ** 2 + (y[0] - y[-1]) ** 2))

    x_s_spline = CubicSpline(ss, x)
    y_s_spline = CubicSpline(ss, y)

    dxds = x_s_spline(ss, 1)
    dyds = y_s_spline(ss, 1)

    dx2d2s = x_s_spline(ss, 2)
    dy2d2s = y_s_spline(ss, 2)
    r = np.sqrt(dx2d2s ** 2 + dy2d2s ** 2)

    return dxds, dyds, ss, dists, r

def preprocess(x, y, vel_base, gain_base):
    # plot the path and plot a subsection of the plot with another colour
    vel_start_index = np.array([650, 1150, 100, 951, 901])
    vel_mid_index = np.array([850, 1250, 0, 0, 0])
    vel_end_index = np.array([900, x.shape[0], 649, 1149, 1025])
    vel_start1_index = 100
    offset_gain = 100
    target_vel = vel_base - 1.3
    target_vel_index_4 = vel_base[4]
    gain = np.ones_like(x)
    gain = gain*gain_base[0]
    target_gain = gain_base + 0.1
    vel = np.ones_like(x)
    vel = vel*vel_base[0]
    for i in range(len(vel_start_index)):
        if i == 0: 
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            
            # an linear intropolation from the base velocity to the target velocity meeting at the middle
            for j in range(len(vel_index)-1):
                gain[vel_index[j]] = target_gain[i] + 0.1
                if j < vel_mid_index[i] - vel_start_index[i]:
                    # vel[vel_index[j]] = vel_base[i] - (vel_base[i] - target_vel[i]) * (j / (vel_mid_index[i] - vel_start_index[i]))
                    vel[vel_index[j]] = target_vel[i]
                    if j > (vel_mid_index[i] - vel_start_index[i]) - offset_gain :
                        pass
                        # gain[vel_index[j]] = target_gain[i]
                else:
                    vel[vel_index[j]] = target_vel[i] + (vel_base[i] - target_vel[i]) * ((j - (vel_mid_index[i] - vel_start_index[i]))/ (vel_end_index[i] - vel_mid_index[i]))
                    # gain[vel_index[j]] = target_gain[i] 
        if i == 2:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            for j in range(len(vel_index)):
                vel[vel_index[j]] = vel_base[i]
        if i == 3:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            for j in range(len(vel_index)):
                vel[vel_index[j]] = vel_base[i]
        if i == 1:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            
            # an linear intropolation from the base velocity to the target velocity meeting at the middle
            for j in range(len(vel_index)):
                # vel[vel_index[j]] = vel_base[i] - (vel_base[i] - target_vel[i]) * (j / (vel_end_index[i] - vel_start_index[i]))
                vel[vel_index[j]] = target_vel[i]
                gain[vel_index[j]] = target_gain[i] + 0.1
           
            for j in range(vel_start1_index):
                vel[j] = target_vel[i] + (vel_base[i] - target_vel[i]) * (j / (vel_start1_index))
                if j < 50:
                    # gain[j] = target_gain[i] + 0.1
                    pass
        if i == 4:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            for j in range(len(vel_index)):
                vel[vel_index[j]] = target_vel_index_4

def preprocess2(x, y, vel_base, vel_target, gain_base, vel_start_index, vel_end_index):
    vel_start1_index = 100
    offset_gain = 100
    target_vel = vel_base - 1.8
    target_vel_index_4 = vel_base[4]
    gain = np.ones_like(x)
    gain = gain*gain_base[1]
    target_gain = gain_base + 0.2
    vel = np.ones_like(x)
    vel = vel*vel_base[0]
    for i in range(len(vel_start_index)):
        if i == 0: 
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            
            # an linear intropolation from the base velocity to the target velocity meeting at the middle
            for j in range(len(vel_index)):
                if j> 20:
                    gain[vel_index[j]] = gain_base[i]
                else:
                    gain[vel_index[j]] = gain_base[i-1] - 0.20
                vel[vel_index[j]] = vel_target[i]
                # else:
                #     vel[vel_index[j]] = target_vel[i] + (vel_base[i] - target_vel[i]) * ((j - (vel_mid_index[i] - vel_start_index[i]))/ (vel_end_index[i] - vel_mid_index[i]))
                    # gain[vel_index[j]] = target_gain[i] 
        if i == 1:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            
            # an linear intropolation from the base velocity to the target velocity meeting at the middle
            for j in range(len(vel_index)):
                vel[vel_index[j]] = vel_target[i] + (vel_base[i] - vel_target[i]) * (j / (vel_end_index[i] - vel_start_index[i]))
           
            # for j in range(vel_start1_index):
            #     vel[j] = target_vel[i] + (vel_base[i] - target_vel[i]) * (j / (vel_start1_index))
            #     if j < 50:
            #         # gain[j] = target_gain[i] + 0.1
            #         pass        
        if i == 2:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            for j in range(len(vel_index)):
                vel[vel_index[j]] = vel_base[i]
        if i == 3:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            for j in range(len(vel_index)):
                vel[vel_index[j]] = vel_base[i] - (vel_base[i] - vel_target[i]) * (j / (vel_end_index[i] - vel_start_index[i]))
        if i == 4:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            for j in range(len(vel_index)):
                vel[vel_index[j]] = vel_target[i]
                gain[vel_index[j]] = gain_base[i]

        if i == 5:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            vel_index2 = np.arange(vel_start_index[i+1], vel_end_index[i+1])
            vel_index = np.concatenate((vel_index, vel_index2))

            for j in range(len(vel_index)):
                vel[vel_index[j]] = vel_target[i] + (vel_base[i] - vel_target[i]) * (j / len(vel_index))

            for j in range(len(vel_index)):
                gain[vel_index[j]] = gain_base[i]   

        if i==6:
            pass

        if i == 7:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            for j in range(len(vel_index)):
                vel[vel_index[j]] = vel_target[i] + (vel_base[i] - vel_target[i]) * (j / len(vel_index))
                gain[vel_index[j]] = gain_base[i]
        if i == 8:
            vel_index = np.arange(vel_start_index[i],  vel_end_index[i])
            for j in range(len(vel_index)):
                vel[vel_index[j]] = vel_base[i]

                if j > (vel_end_index[i] - 20 - vel_start_index[i]):
                    gain[vel_index[j]] = gain_base[i] - 0.20
                elif (j > (vel_end_index[i] - 230 - vel_start_index[i]))  and (j <= (vel_end_index[i] - 20 - vel_start_index[i])):
                    gain[vel_index[j]] = gain_base[i] + 0.15
                else:
                    gain[vel_index[j]] = gain_base[i]
    
        
        if i == 9:
            vel_index = np.arange(vel_start_index[i], vel_end_index[i])
            for j in range(len(vel_index)):
                vel[vel_index[j]] = vel_base[i] - (vel_base[i] - vel_target[i]) * (j / len(vel_index))
                # if j< 50:
                #     gain[vel_index[j]] = gain_base[i]



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Velocity')
    # plt.axis('equal')
    ax.plot(x, y, vel)
    print(gain[gain.shape[0]-10:], gain[:10])
    # gain_start_index = np.array([850, 1250, 0])
    # gain_end_index = np.array([950, x.shape[0]+1, vel_start1_index])
    # target_gain = gain_base + 0.1
    # gain = np.ones_like(x)
    # gain = gain*gain_base[0]
    # for i in range(len(gain_start_index)):
    #     gain_index = np.arange(gain_start_index[i], gain_end_index[i])
        
    #     # an linear intropolation from the base velocity to the target velocity meeting at the middle
    #     for j in range(len(gain_index)-1):
    #             gain[gain_index[j]] = target_gain[i]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Gain')
    # plt.axis('equal')
    ax.plot(x, y, gain)
    # plt.axis('equal')
    plt.show()

    return vel, gain

# Function to read csv file
def read_csv(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            x.append(float(line.split(',')[0]))
            y.append(float(line.split(',')[1]))
    return x, y

# Function to plot the path
def plot_path(x, y):
    ban_aeb_range = [[0, 250], [250, 420], [420, 720], [850, 900], [900, 1100], [1100, 1250]]
    plt.plot(x, y)

    for i in range(len(ban_aeb_range)):
        start_ind = ban_aeb_range[i][0]
        end_ind = ban_aeb_range[i][1]
        new_xs = x[start_ind:end_ind + 1]
        new_ys = y[start_ind:end_ind + 1]
        plt.plot(new_xs, new_ys, 'o')
    
    plt.show()

# plot 3D plot with theta as z axis
def plot_3d(x, y, theta):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, theta)
    plt.show()

# Function to save x,y and theta to csv file
def save_csv(x, y, theta, vel, gain):
    with open('traj_race_cl_0_3_theta.csv', 'w') as f:
        for i in range(len(x)):
            f.write(str(x[i]) + ',' + str(y[i]) + ',' + str(theta[i]) + ',' + str(vel[i]) + ',' + str(gain[i]) + '\n')

# main
if __name__ == '__main__':
    x, y = read_csv('traj_race_cl_70.csv')
    theta = compute_theta(x, y)
    x = np.array(x)
    y = np.array(y)

    # plot the path and plot a subsection of the plot with another colour
    vel_start_index = np.array([600, 860, 951, 1125, 1200, 1360,        0, 100,  300, 525])
    # vel_mid_index = np.array([850, 1250, 0, 0, 0])
    vel_end_index = np.array([860, 951, 1125, 1200, 1360, x.shape[0], 100,  300, 525, 600])

    vel_base = np.array(  [5.5,  5.5,  5.5,  5.5,  5.5,  5.5,   4.5,  6.5, 6.5,  6.5])
    vel_target = np.array([4.1,  4.1,  4.2,  4.2,  4.1,  4.1,   4.2, 5.5, 4.2,  4.2])
    gain_base = np.array( [0.70, 0.45, 0.55, 0.45, 0.65, 0.65,  0.65, 0.65, 0.65, 0.65])
    vel, gain = preprocess2(x, y, vel_base, vel_target, gain_base, vel_start_index, vel_end_index)
    # vel = compute_vel(x, y)
    # gain = compute_gain(x, y)
    # dxds, dyds, ss, dists, r = compute_derivative(x, y)
    plot_path(x, y)
    # plot_3d(x, y, theta)
    save_csv(x, y, theta, vel, gain)