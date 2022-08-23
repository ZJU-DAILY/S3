import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def points2meter(points):
    rtn = []
    for p in points:
        lon_meter, lat_meter = lonlat2meters(lon=p[1], lat=p[0])
        rtn.append([lat_meter,lon_meter,p[2]])
    return rtn

def to_traj(file):
    traj = []
    f = open(file)
    for line in f:
        temp = line.strip().split(' ')
        if len(temp) < 3:
            continue
        traj.append([float(temp[0]), float(temp[1]), int(float(temp[2]))])
    f.close()
    return traj

def sed_op(segment):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0
    else:
        #print('segment', segment)
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1,len(segment)-1):
            syn_time = segment[i][2]
            time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
            syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
            syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
            e = max(e, np.linalg.norm(np.array([segment[i][0],segment[i][1]]) - np.array([syn_x,syn_y])))
        #print('segment error', e)
        return e

def sed_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            error = max(error, sed_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def ped_op(segment):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1,len(segment)-1):
            pm = segment[i]
            A = pe[1] - ps[1]
            B = ps[0] - pe[0]
            C = pe[0] * ps[1] - ps[0] * pe[1]
            if A == 0 and B == 0:
                e = max(e, 0.0)
            else:
                e = max(e, abs((A * pm[0] + B * pm[1] + C)/ np.sqrt(A * A + B * B)))
        #print('segment error', e)
        return e

def ped_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            error = max(error, ped_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def angle(v1):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    angle1 = math.atan2(dy1, dx1)
    if angle1 >= 0:
        return angle1
    else:
        return 2*math.pi + angle1
    
def dad_op(segment):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        theta_0 = angle([ps[0],ps[1],pe[0],pe[1]])
        for i in range(0,len(segment)-1):
            pm_0 = segment[i]
            pm_1 = segment[i+1]
            theta_1 = angle([pm_0[0],pm_0[1],pm_1[0],pm_1[1]])
            e = max(e, min(abs(theta_0 - theta_1), 2*math.pi - abs(theta_0 - theta_1)))
        #print('segment error', e)
        return e

def dad_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            error = max(error, dad_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def get_point(ps, pe, segment, index):
    syn_time = segment[index][2]
    time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
    syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
    syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
    return [syn_x, syn_y], syn_time

def speed_op(segment):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(0,len(segment)-1):
            p_1, t_1 = get_point(ps, pe, segment, i)
            p_2, t_2 = get_point(ps, pe, segment, i+1)
            time = 1 if t_2 - t_1 == 0 else abs(t_2-t_1)
            est_speed = np.linalg.norm(np.array(p_1) - np.array(p_2))/time
            rea_speed = np.linalg.norm(np.array([segment[i][0], segment[i][1]]) - np.array([segment[i+1][0], segment[i+1][1]]))/time
            e = max(e, abs(est_speed - rea_speed))
        #print('segment error', e)
        return e

def speed_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            error = max(error, speed_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def draw_sed_op(segment):
    if len(segment) <= 2:
        #print('segment error', 0.0)
        return 0.0, segment[0], segment[0], segment[0], segment[0]
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1,len(segment)-1):
            syn_time = segment[i][2]
            time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
            syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
            syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
            t = np.linalg.norm(np.array([segment[i][0],segment[i][1]]) - np.array([syn_x,syn_y]))
            if t >= e:
                e = t
                e_points = segment[i]
                syn = [syn_x, syn_y]
        #print('segment error', e)
        return e, e_points, ps, pe, syn
    
def draw_error(ori_traj, sim_traj, label):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            #print(start, c)
            if label == 'sed':
                e,  e_points, ps, pe, syn = draw_sed_op(ori_traj[start: c+1])
                if e > error:
                    error = e
                    error_points = e_points
                    error_syn = syn
                    error_left = ps
                    error_right = pe
            start = c
    return error, error_points, error_left, error_right, error_syn

def draw(ori_traj, sim_traj, label='sed'):
    error, error_points, error_left, error_right, error_syn = draw_error(ori_traj, sim_traj, label)
    pdf = PdfPages('vis_rlts_geo_sed_online.pdf')
    plt.figure(figsize=(10.5/2,6.8/2)) 
    plt.plot(np.array(ori_traj)[:,0],np.array(ori_traj)[:,1],color="blue", linewidth=0.7, label='raw traj')
    plt.scatter(np.array(sim_traj)[:,0],np.array(sim_traj)[:,1],color="red", s=20)
    plt.plot(np.array(sim_traj)[:,0],np.array(sim_traj)[:,1], '--', color="red", linewidth=0.5, label='simplified traj')
    #plt.scatter(error_points[0],error_points[1],color="black", s=30, marker='s', label='maximal error point')
    plt.plot([error_points[0],error_syn[0]],[error_points[1],error_syn[1]], '--', color="black", label='SED')
    plt.plot([error_left[0],error_right[0]],[error_left[1],error_right[1]], color="green", linewidth=2, label='anchor seg')
    plt.title('simplified traj length: '+str(len(sim_traj)))
    plt.legend(loc='best', prop = {'size': 9})
    pdf.savefig()
    pdf.close()
    return error

if __name__ == '__main__':
    print('The required tools are implemented here!')