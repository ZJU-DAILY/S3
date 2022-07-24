import numpy as np
import data_utils as F
import copy
import heapq
from heapq import heappush, heappop, _siftdown, _siftup
import matplotlib.pyplot as plt
import math
import pickle

class TrajComp():
    def __init__(self, path, amount, a_size, s_size):
        self.n_actions = a_size
        self.n_features = s_size
        self._load(path, amount)

    def _load(self, path, amount):
        self.ori_traj_set = []
        for num in range(amount):
            self.ori_traj_set.append(F.to_traj(path + str(num)))
            
    def read(self, p, episode):
        self.F_ward[self.link_tail] = [0.0, p]
        self.B_ward[p] = [0.0, self.link_tail]
        s = self.B_ward[self.link_tail][1]
        m = self.link_tail
        e = self.F_ward[self.link_tail][1]
        self.err_record[(s, e)] = F.sed_op(self.ori_traj_set[episode][s: e + 1])
        self.F_ward[m][0] = self.err_record[(s, e)]
        self.B_ward[m][0] = self.err_record[(s, e)]
        heapq.heappush(self.heap, (self.F_ward[m][0], m))# save (state_value, point index of ori traj)
        self.link_tail = p
    
    def reset(self, episode, buffer_size):
        self.rw = 0.0
        self.INX = 0
        self.heap = []
        self.last_error = 0.0
        self.current = 0.0
        self.c_left = 0
        self.c_right = 0
        #self.copy_traj = copy.deepcopy(self.ori_traj_set[episode])
        self.start = {}
        self.end = {}
        self.err_seg = {}
        self.err_record = {}
        self.steps = len(self.ori_traj_set[episode])
        self.F_ward = {} # save (state_value, next_point)
        self.B_ward = {} # save (state_value, last_point)
        self.F_ward[0] = [0.0, 1]
        self.B_ward[1] = [0.0, 0]
        self.link_head = 0
        self.link_tail = 1
        for i in range(2, buffer_size + 1):
            self.read(i, episode)
        t = heapq.nsmallest(self.n_features, self.heap)
        if len(t) < self.n_features:
            self.check = [t[0][1],t[0][1],t[1][1]]
            self.state = [t[0][0],t[0][0],t[1][0],t[0][0],t[0][0]]
        else:
            self.check = [t[0][1], t[1][1],t[2][1]]
            if buffer_size + 4 <= self.steps:
                J1 = F.sed_op(self.ori_traj_set[episode][buffer_size: buffer_size + 3])
                J2 = F.sed_op(self.ori_traj_set[episode][buffer_size: buffer_size + 4])
                self.state = [t[0][0], t[1][0],t[2][0], J1, J2]
            else:
                self.state = [t[0][0], t[1][0],t[2][0], t[0][0], t[0][0]]
#        self.check = [self.heap[0][1], self.heap[1][1]]
#        self.state = [self.heap[0][0], self.heap[1][0]] # state = (min_non_b; min_b; current)
        #print('len, obs, heap and state', len(self.heap), self.observation, self.heap, self.state)
        return self.steps, np.array(self.state).reshape(1, -1)           
        
    def reward_update(self, episode, rem, label=''):
        if label == 'skip':
            a = rem[0]
            b = rem[1]
            self.start[a] = b
            self.end[b] = a
            NOW = F.sed_op(self.ori_traj_set[episode][a: b + 1])
            self.err_seg[(a,b)] = self.err_record[(a,b)] = NOW
            if NOW >= self.last_error:
                self.current = NOW
                self.current_left, self.current_right = a, b
            return
        
        if (rem not in self.start) and (rem not in self.end):
            #print('interval insert')
            #f.write('interval insert\n')
            a = self.B_ward[rem][1]
            b = self.F_ward[rem][1]
            self.start[a] = b
            self.end[b] = a
            NOW = self.err_record[(a,b)]
            self.err_seg[(a,b)] = NOW
            if NOW >= self.last_error:
                self.current = NOW
                self.current_left, self.current_right = a, b
        
        elif (rem in self.start) and (rem not in self.end):
            #print('interval expand left')
            a = self.B_ward[rem][1]
            b = rem
            c = self.start[rem]
            BEFORE = self.err_record[(b, c)]
            NOW = self.err_record[(a, c)]
            del self.err_seg[(b,c)]
            self.err_seg[(a,c)] = NOW
            
            if  math.isclose(self.last_error,BEFORE):
                if NOW >= BEFORE:
                    #print('interval expand left_case1')
                    #f.write('interval expand left_case1 \n')
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #print('interval expand left_case2')
                    #f.write('interval expand left_case2 \n')
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
            else:
                #print('interval expand left_case3')
                #f.write('interval expand left_case3 \n')
                if NOW >= self.last_error:
                    self.current = NOW
                    self.current_left, self.current_right = a, c
            self.end[c] = a
            self.start[a] = c
            del self.start[b]
            
        # interval expand right
        elif (rem not in self.start) and (rem in self.end):
            #print('interval expand right')
            a = self.end[rem]
            b = rem
            c = self.F_ward[rem][1]
            BEFORE = self.err_record[(a, b)]
            NOW = self.err_record[(a, c)]
            del self.err_seg[(a,b)]
            self.err_seg[(a,c)] = NOW
            if math.isclose(self.last_error,BEFORE):
                if NOW >= BEFORE:
                    #print('interval expand right_case1')
                    #f.write('interval expand right_case1 \n')
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #print('interval expand right_case2')
                    #f.write('interval expand right_case2 \n')
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
            else:
                #print('interval expand right_case3')
                #f.write('interval expand right_case3 \n')
                if NOW >= self.last_error:
                    self.current = NOW
                    self.current_left, self.current_right = a, c
            self.start[a] = c
            self.end[c] = a
            del self.end[b]
        
        # interval merge
        elif (rem in self.start) and (rem in self.end):
            #print('interval merge')
            b = rem
            a = self.end[b]
            c = self.start[b]
            # get values quickly
            BEFORE_1 = self.err_record[(a, b)]
            BEFORE_2 = self.err_record[(b, c)]
            NOW = self.err_record[(a, c)]
            del self.err_seg[(a,b)]
            del self.err_seg[(b,c)]
            self.err_seg[(a,c)] = NOW            
            if math.isclose(self.last_error,BEFORE_1):
                if NOW >= BEFORE_1:
                    #print('interval merge_case1')
                    #f.write('interval merge_case1 \n')
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #print('interval merge_case2')
                    #f.write('interval merge_case2 \n')
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
                    
            elif math.isclose(self.last_error,BEFORE_2):
                if NOW >= BEFORE_2:
                    #print('interval merge_case3')
                    #f.write('interval merge_case3 \n')
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #print('interval merge_case4')
                    #f.write('interval merge_case4 \n')
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
            else:
                #print('interval merge_case5')
                #f.write('interval merge_case5 \n')
                if NOW >= self.last_error:
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                    
            self.start[a] = c
            self.end[c] = a
            del self.start[b]
            del self.end[b]
        else:
            print('Here is a bug!!!')
    
    def delete_heap(self, heap, nodeValue):
        leafValue = heap[-1]
        i = heap.index(nodeValue)
        if nodeValue == leafValue:
            heap.pop(-1)
        elif nodeValue <= leafValue: # similar to heappop
            heap[i], heap[-1] = heap[-1], heap[i]
            minimumValue = heap.pop(-1)
            if heap != []:
                _siftup(heap, i)
        else: # similar to heappush
            heap[i], heap[-1] = heap[-1], heap[i]
            minimumValue = heap.pop(-1)
            _siftdown(heap, 0, i)
        
    def step(self, episode, action, index, done, label='T'):        
        # update state and compute reward
        #print('self.F_ward', self.F_ward)
        #print('check, state, heap', self.check, self.state, self.heap)
        if action >= len(self.check):
            rem = self.check[0]
        else:
            rem = self.check[action] # point index in ori traj
        #print('remove point index and value', self.state, rem, self.F_ward[rem][0])
        NEXT_P = self.F_ward[rem][1]
        NEXT_V = self.B_ward[NEXT_P][0]
        LAST_P = self.B_ward[rem][1]
        LAST_V = self.F_ward[LAST_P][0]

        if LAST_P > self.link_head:
            self.delete_heap(self.heap, (LAST_V, LAST_P))
            #s = self.ori_traj_set[episode][self.B_ward[LAST_P][1]]
            #m1 = self.ori_traj_set[episode][LAST_P]
            #m2 = self.ori_traj_set[episode][rem]
            #e = self.ori_traj_set[episode][NEXT_P]
            self.err_record[(self.B_ward[LAST_P][1], NEXT_P)] = F.sed_op(self.ori_traj_set[episode][self.B_ward[LAST_P][1]: NEXT_P + 1])
            #self.err_record[(self.B_ward[LAST_P][1], NEXT_P)] = F.sed_op([s,m1,m2,e])
            self.F_ward[LAST_P][0] = self.err_record[(self.B_ward[LAST_P][1], NEXT_P)]
            self.B_ward[LAST_P][0] = self.err_record[(self.B_ward[LAST_P][1], NEXT_P)]
            heapq.heappush(self.heap, (self.F_ward[LAST_P][0], LAST_P))
        if NEXT_P < self.link_tail:
            self.delete_heap(self.heap, (NEXT_V, NEXT_P))
            #s = self.ori_traj_set[episode][LAST_P]
            #m1 = self.ori_traj_set[episode][rem]
            #m2 = self.ori_traj_set[episode][NEXT_P]
            #e = self.ori_traj_set[episode][self.F_ward[NEXT_P][1]]
            self.err_record[(LAST_P, self.F_ward[NEXT_P][1])] = F.sed_op(self.ori_traj_set[episode][LAST_P: self.F_ward[NEXT_P][1] + 1])
            #self.err_record[(LAST_P, self.F_ward[NEXT_P][1])] = F.sed_op([s,m1,m2,e])
            self.F_ward[NEXT_P][0] = self.err_record[(LAST_P, self.F_ward[NEXT_P][1])]
            self.B_ward[NEXT_P][0] = self.err_record[(LAST_P, self.F_ward[NEXT_P][1])]
            heapq.heappush(self.heap, (self.F_ward[NEXT_P][0], NEXT_P))
        
        #self.copy_traj.remove(self.ori_traj_set[episode][rem])
        if  label == 'T':
            self.reward_update(episode, rem)
            '''
            self.copy_traj.remove(self.ori_traj_set[episode][rem])
            _,  self.current = F.sed_error(self.ori_traj_set[episode], self.copy_traj)
            '''
            self.rw = self.last_error - self.current
            self.last_error = self.current
            #print('self.current',self.current)
        
        self.F_ward[LAST_P][1] = NEXT_P
        self.B_ward[NEXT_P][1] = LAST_P
        self.delete_heap(self.heap, (self.F_ward[rem][0], rem))
        del self.F_ward[rem]
        del self.B_ward[rem]     
        
        if not done:
            if action >= len(self.check):
                self.INX = min(index + 2 + action - len(self.check), len(self.ori_traj_set[episode]) - 1)
                self.read(self.INX, episode)
                if  label == 'T':
                    self.reward_update(episode, [index, self.INX], 'skip')
                    '''
                    for skip in range(index + 1, self.INX):
                        self.copy_traj.remove(self.ori_traj_set[episode][skip])
                    _,  self.current = F.sed_error(self.ori_traj_set[episode], self.copy_traj)
                    '''
                    self.rw += self.last_error - self.current
                    self.last_error = self.current
            else:
                self.read(index + 1, episode)
            t = heapq.nsmallest(self.n_features, self.heap)
            if len(t) < self.n_features:
                self.check = [t[0][1],t[0][1],t[1][1]]
                self.state = [t[0][0],t[0][0],t[1][0],t[0][0],t[0][0]]
            else:
                self.check = [t[0][1], t[1][1],t[2][1]]
                if self.INX + 4 <= self.steps:
                    J1 = F.sed_op(self.ori_traj_set[episode][self.INX: self.INX + 3])
                    J2 = F.sed_op(self.ori_traj_set[episode][self.INX: self.INX + 4])
                    self.state = [t[0][0], t[1][0],t[2][0], J1, J2]
                else:
                    self.state = [t[0][0],t[1][0],t[2][0],t[0][0],t[0][0]]
#            self.check = [self.heap[0][1], self.heap[1][1]]
#            self.state = [self.heap[0][0], self.heap[1][0]]
        #f.write('--->'+str(rw)+'\n')
        #self.state = [max(self.heap[0][0] - self.current, 0.0)]
        #cannot remove the starting and ending
#        if self.current_left == self.link_head:
#            self.check.append(self.current_right)
#            self.state.append(self.B_ward[self.current_right][0])
#        elif self.current_right == self.link_tail:
#            self.check.append(self.current_left)
#            self.state.append(self.F_ward[self.current_left][0])
#        elif self.F_ward[self.current_left][0] < self.B_ward[self.current_right][0]:
#            self.check.append(self.current_left)
#            self.state.append(self.F_ward[self.current_left][0])
#        else:
#            self.check.append(self.current_right)
#            self.state.append(self.B_ward[self.current_right][0])
            
        #self.state.append(self.current)
        
        #self.state[1] = self.state[1] - self.current

        #print('check and state', self.check, self.state)
        return np.array(self.state).reshape(1, -1), self.rw
    
    def output(self, episode, label = 'T'):
        if label == 'V-VIS':
            start = 0
            sim_traj = []
            while start in self.F_ward:
                sim_traj.append(self.ori_traj_set[episode][start])
                start = self.F_ward[start][1]
            sim_traj.append(self.ori_traj_set[episode][start])
            _, final_error = F.sed_error(self.ori_traj_set[episode], sim_traj)
            print('Validation at episode {} with error {}'.format(episode, final_error))
            #for visualization, 'sed' is by default, if you want to draw other errors by revising the codes in data_utils.py correspondingly.
            F.draw(self.ori_traj_set[episode], sim_traj) 
            return final_error
        if label == 'V':
            start = 0
            sim_traj = []
            while start in self.F_ward:
                sim_traj.append(self.ori_traj_set[episode][start])
                start = self.F_ward[start][1]
            sim_traj.append(self.ori_traj_set[episode][start])
            _, final_error = F.sed_error(self.ori_traj_set[episode], sim_traj)
            return final_error
        if label == 'T':
            print('Training at episode {} with error {}'.format(episode, self.current))
            return self.current
