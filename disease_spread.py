"""
Animation of Disease Spread (adapted from Jake Vanderplas graphic)
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import math

class ParticleBox:
    """Orbits class
    
    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax, wall1_x1, wall2_x2, wall1_ymini, box_2_xmax, box_2_ymin, box_2_ymax, box_3_xmin, box_3_ymin, box_3_ymax]
    """
    def __init__(self,
                 init_state = [[1, 0, 0, -1],
                               [-0.5, 0.5, 0.5, 0.5],
                               [-0.5, -0.5, -0.5, 0.5]],
                 bounds = [-2, 2, -2, 2, -1, -0.333, 0, 0.5, -1, -0.7, -0.5, -1.5, -1.3],
                 size = 0.04,
                 M = 0.05,
                 G = 9.8,
                 infected = [0],
                 infected_states = []):

        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G
        self.infected = infected
        self.infected_states =infected_states
    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        
        # update positions

        self.state[:, :2] += dt * self.state[:, 2:]
        self.infected_states[:, :2] += dt * self.infected_states[:, 2:]
        to_remove = []
        #add_infected_ppl
        for idx in range(self.infected_states.shape[0]):
            for idx_2  in range(self.state.shape[0]):
                if (math.sqrt((self.state[idx_2,0] - self.infected_states[idx,0])**2 + (self.state[idx_2,1] - self.infected_states[idx,1])**2) < 0.2):
                    if not (idx_2 in to_remove):
                        to_remove.append(idx_2)
                        self.infected.append(idx_2)
                        #print(idx_2)
        infected_states_updated = [] #(self.infected_states).copy().tolist()
        new_self_states = []
        
        #updated infected and non-infected states. If within the new radius, we add to infected, else, we add to the updated non-infected states
        for idx in range(self.state.shape[0]):
            if (idx not in to_remove):
                new_self_states.append(self.state[idx,:])
            else:
                #if (self.state[idx,:] not in self.infected_states):
                infected_states_updated.append(self.state[idx,:])
        self.infected_states = np.asarray(infected_states_updated)
        self.states= np.asarray(new_self_states)

        print("NON INFECTED: " + str(self.state.shape[0]))
        print("INFECTED: " + str(self.infected_states.shape[0])) 
        #updated_states = []
        #for idx in range(self.init_state.shape[0]):
        #    if idx not in self.infected:
        #        updated_states.append(self.state[idx, :])
        #self.state = np.asarray(updated_states)

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 2 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 2:] = v_cm - v_rel * m1 / (m1 + m2) 

        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)
        crossed_wall_x1 = (self.state[:,0] >  self.bounds[4] - self.size)
        crossed_wall_x2 = (self.state[:,0] < self.bounds[5] + self.size)
        crossed_wall_y = (self.state[:,1] > self.bounds[6] - self.size)
        crossed_wall2_x = (self.state[:,0] > self.bounds[7] - self.size)
        crossed_wall2_ymin = (self.state[:,1] >  self.bounds[8] - self.size)
        crossed_wall2_ymax = (self.state[:,1] < self.bounds[9] + self.size)
        crossed_wall3_x = (self.state[:,0] < self.bounds[10] + self.size)
        crossed_wall3_ymin = (self.state[:,1] > self.bounds[11] - self.size)
        crossed_wall3_ymax = (self.state[:,1] < self.bounds[12] + self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        #self.state[crossed_wall_x1 & crossed_wall_y, 0] = self.bounds[4] + self.size
        #self.state[crossed_wall_x2 & crossed_wall_y, 0] = self.bounds[5] - self.size

        #self.state[crossed_wall_y, 1] = self.bounds[6] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1
        self.state[(crossed_wall_x1 & crossed_wall_x2)  & crossed_wall_y, 2] *= -1
        #self.state[crossed_wall_x2  &  crossed_wall_y, 2] *= -1
        #self.state[(crossed_wall_x2 & crossed_wall_x1) & crossed_wall_y, 3] *= -1

        self.state[(crossed_wall2_ymax & crossed_wall2_ymin) & crossed_wall2_x, 2] *= -1
        self.state[(crossed_wall2_ymax & crossed_wall2_ymin) & crossed_wall2_x, 3] *= -1
        self.state[(crossed_wall3_ymax & crossed_wall3_ymin) & crossed_wall3_x, 2] *= -1
        self.state[(crossed_wall3_ymax & crossed_wall3_ymin) & crossed_wall3_x, 3] *= -1
        # add gravity
        #self.state[:, 3] -= self.M * self.G * dt

        crossed_x1 = (self.infected_states[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.infected_states[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.infected_states[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.infected_states[:, 1] > self.bounds[3] - self.size)
        crossed_wall_x1 = (self.infected_states[:,0] >  self.bounds[4] - self.size)
        crossed_wall_x2 = (self.infected_states[:,0] < self.bounds[5] + self.size)
        crossed_wall_y = (self.infected_states[:,1] > self.bounds[6] - self.size)
        crossed_wall2_x = (self.infected_states[:,0] > self.bounds[7] - self.size)
        crossed_wall2_ymin = (self.infected_states[:,1] >  self.bounds[8] - self.size)
        crossed_wall2_ymax = (self.infected_states[:,1] < self.bounds[9] + self.size)
        crossed_wall3_x = (self.infected_states[:,0] < self.bounds[10] + self.size)
        crossed_wall3_ymin = (self.infected_states[:,1] > self.bounds[11] - self.size)
        crossed_wall3_ymax = (self.infected_states[:,1] < self.bounds[12] + self.size)

        self.infected_states[crossed_x1, 0] = self.bounds[0] + self.size
        self.infected_states[crossed_x2, 0] = self.bounds[1] - self.size

        #self.infected_states[crossed_wall_x1 & crossed_wall_y, 0] = self.bounds[4] + self.size
        #self.infected_states[crossed_wall_x2 & crossed_wall_y, 0] = self.bounds[5] - self.size

        #self.infected_states[crossed_wall_y, 1] = self.bounds[6] - self.size

        self.infected_states[crossed_y1, 1] = self.bounds[2] + self.size
        self.infected_states[crossed_y2, 1] = self.bounds[3] - self.size

        self.infected_states[crossed_x1 | crossed_x2, 2] *= -1
        self.infected_states[crossed_y1 | crossed_y2, 3] *= -1
        self.infected_states[(crossed_wall_x1 & crossed_wall_x2)  & crossed_wall_y, 2] *= -1
        #self.infected_states[crossed_wall_x2  &  crossed_wall_y, 2] *= -1
        #self.infected_states[(crossed_wall_x2 & crossed_wall_x1) & crossed_wall_y, 3] *= -1

        self.infected_states[(crossed_wall2_ymax & crossed_wall2_ymin) & crossed_wall2_x, 2] *= -1
        self.infected_states[(crossed_wall2_ymax & crossed_wall2_ymin) & crossed_wall2_x, 3] *= -1
        self.infected_states[(crossed_wall3_ymax & crossed_wall3_ymin) & crossed_wall3_x, 2] *= -1
        self.infected_states[(crossed_wall3_ymax & crossed_wall3_ymin) & crossed_wall3_x, 3] *= -1
#------------------------------------------------------------
# set up initial state
np.random.seed(2)

new_boxes = []
bounds = [-2, 2, -2, 2, -1, -0.333, 0]

#box = ParticleBox(init_state, size=0.04)
#dt = 1. / 30 # 30fps


#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))

# particles holds the locations of the particles
particles, = ax.plot([], [], 'bo', ms=6)
particles_infected, = ax.plot([], [], 'ro', ms=6)

# rect is the box edge
rect = plt.Rectangle(bounds[::2],
                     bounds[1] - bounds[0],
                     bounds[3] - bounds[2],
                     ec='none', lw=2, fc='none')
width_box_1 = (bounds[1]-bounds[0])/6
height_box_1 = 2
box_1= plt.Rectangle([-1,0], width_box_1, height_box_1)
box_2= plt.Rectangle([0.5, -1], 1.5, 0.3)
box_3= plt.Rectangle([-2,-1.5], 1.5, 0.2)
ax.add_patch(box_1)
ax.add_patch(box_2)
ax.add_patch(box_3)
ax.add_patch(rect)

new_boxes.append([-1,-1+width_box_1, 0, 0 + height_box_1])
new_boxes.append([0.5, 2,-1, -0.7])
new_boxes.append([-2, -0.5, -1.5, -1.3])
num_removed = 0
#to eliminate any balls that were initialized within the box

init_state = -0.5 + np.random.random((500, 4)) #this should be changed to -0.5
init_state[:, :2] *= 3.9
idx_restricted = []

for row in init_state:
    for box_current in new_boxes:
        x1 = box_current[0]
        x2 = box_current[1]
        y1 = box_current[2]
        y2 = box_current[3]
        if (row[0] >= x1 and row[0] <= x2 and row[1] >= y1 and row[1] <= y2):
            as_list = row.tolist()
            idx_restricted.append(as_list)

final_init_state = []
for row in init_state:
    row =row.tolist()
    if not (row in idx_restricted):
        final_init_state.append(row)

to_add_to_infected = final_init_state[0]
final_infected_state = [to_add_to_infected]
final_infected_state = np.asarray(final_infected_state)
final_init_state = np.asarray(final_init_state)

box = ParticleBox(init_state = final_init_state, infected_states = final_infected_state, size=0.02)
dt = 1./100

def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    particles_infected.set_data([],[])
    rect.set_edgecolor('none')
    return particles, particles_infected, rect

def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)
    to_add_to_infected = []
    for elem in box.infected:
        to_add_to_infected.append(box.init_state[elem, :].tolist())
    to_add_to_infected = np.asarray(to_add_to_infected)
    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    
    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    particles_infected.set_data(box.infected_states[:,0], box.infected_states[:,1])
    particles_infected.set_markersize(ms)
    return particles, particles_infected, rect

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=10, blit=True, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
