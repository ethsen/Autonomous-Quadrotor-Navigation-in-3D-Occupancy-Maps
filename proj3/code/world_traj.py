import numpy as np

from .graph_search import graph_search

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
      
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.5
        self.traj,_ = graph_search(world,self.resolution,self.margin,start,goal,astar= True)
        if self.traj is None:
            self.resolution = np.array([0.15, 0.15, 0.25])
            self.margin = 0.25
            self.traj,_ = graph_search(world,self.resolution,self.margin,start,goal,astar= True)

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = self.traj
        #self.points = rdp(self.points,0.1)
        #self.points = self.rdp(self.points)

        self.goal = goal

        
        self.path = self.points
        #self.points = self.simplify_path(self.points)
        self.fast = []
        for k in range(len(self.points)-1):
            dis = self.points[k,:] - self.points[k+1,:]
            if np.linalg.norm(dis) > 1:#m
                self.fast.append(True)
            else: 
                self.fast.append(False)
        self.fast.append(False)
        self.fast = np.array(self.fast)
        #self.points = np.vstack([self.points,np.array(goal)])
        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.
        self.points[0] = start
        self.points[-1] = goal
        self.velo = 2 #m/s
        self.currSeg = 0
        self.dis = []
        self.segments =[]
        self.unitVecs = []
        self.start()
        self.coeffs = self.computeCoefs()

    def start(self):
        for i in range(len(self.points)-1):
            if self.fast[i]:
                velo = 3 #m/s
            else:
                velo = 4 #m/s
            dis = np.linalg.norm(self.points[i+1] - self.points[i])
            duration = dis/velo
            unitVecs = (self.points[i+1]-self.points[i])/ dis
            self.dis.append(dis)
            self.segments.append(duration)
            self.unitVecs.append(unitVecs)

    def updateSegment(self,time):
        while np.sum(self.segments[:self.currSeg+1]) < time:
            self.currSeg += 1
            if self.fast[self.currSeg]:
                self.velo = 3 #m/s
            else: 
                self.velo = 4#m/s

    def computeCoefs(self):
        coeffs_col = []
        for i in range(len(self.segments)-2):
            a,b = self.constructArr(self.points[i],self.points[i+2],self.points[i+1],i)
            b_x = b[:,0]
            b_y = b[:,1]
            b_z = b[:,2]
            c_x = np.linalg.solve(a,b_x)[:6]
            c_y = np.linalg.solve(a,b_y)[:6]
            c_z = np.linalg.solve(a,b_z)[:6]
            coeffs = np.column_stack([c_x,c_y,c_z])
            coeffs_col.append(coeffs)
        
        return coeffs_col
        
    def constructArr(self,p1,p2,p3,seg):
        t_1 = np.sum(self.segments[:seg+1])
        t_2 = np.sum(self.segments[:seg + 2])
        a = np.array([[0,0,0,0,0,1, 0,0,0,0,0,0],
                      [0,0,0,0,1,0, 0,0,0,0,0,0],
                      [0,0,0,2,0,0, 0,0,0,0,0,0],
                      [0,0,0,0,0,0, t_2**5,t_2**4,t_2**3,t_2**2,t_2,1],
                      [0,0,0,0,0,0, 5*t_2**4,4*t_2**3,3*t_2**2,2*t_2,1,0,],
                      [0,0,0,0,0,0, 20* t_2**3,12*t_2**2,6*t_2,2,0,0],
                      [t_1**5,t_1**4,t_1**3,t_1**2,t_1,1, 0,0,0,0,0,0],#Int positions
                      [0,0,0,0,0,0, 0,0,0,0,0,1],# Int positions
                      [5*t_1**4,4*t_1**3,3*t_1**2,2*t_1,1,0, 0,0,0,0,-1,0],
                      [20* t_1**3,12*t_1**2,6*t_1,2,0,0, 0,0,0,-2,0,0],
                      [60*t_1**2,24*t_1,0,0,0,0, 0,0,-6,0,0,0],
                      [120 * t_1,24,0,0,0,0, 0,-24,0,0,0,0]])
        
        p1_dot = self.velo * self.unitVecs[seg]
        p1_ddot = np.array([0,0,0])
        p2_dot = self.velo * self.unitVecs[seg+2]
        p2_ddot = np.array([0,0,0])

        b = np.vstack([p1,p1_dot,p1_ddot,p2,p2_dot,p2_ddot,p3,p3,np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])])

        return a,b
    
    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0
                
        
        if t > np.sum(self.segments):
            x = self.goal
            flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                    'yaw':yaw, 'yaw_dot':yaw_dot}
            return flat_output
        
        self.updateSegment(t)
        if self.currSeg >= len(self.segments)-2:
            x = self.goal
            flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                    'yaw':yaw, 'yaw_dot':yaw_dot}
            return flat_output
        else: 
            coeffs = self.coeffs[self.currSeg]

            time = np.array([t**5, t**4, t**3, t**2,t,1])
            t_dot = np.array([5* t**4, 4*t**3, 3* t**2, 2* t,1,0])
            t_ddot = np.array([20* t**3, 12*t**2, 6* t, 2,0,0])
            p_x = time @ coeffs[:,0].T
            p_y = time @ coeffs[:,1].T
            p_z = time @ coeffs[:,2].T

            p_x_dot = t_dot @ coeffs[:,0].T
            p_y_dot = t_dot @ coeffs[:,1].T
            p_z_dot = t_dot @ coeffs[:,2].T

            p_x_ddot = t_ddot @ coeffs[:,0].T
            p_y_ddot = t_ddot @ coeffs[:,1].T
            p_z_ddot = t_ddot @ coeffs[:,2].T

            x = np.array([p_x,p_y,p_z])
            x_dot = np.array([p_x_dot,p_y_dot,p_z_dot])
            x_ddot = np.array([p_x_ddot,p_y_ddot,p_z_ddot])
            #yaw = np.arctan2(x_dot[1], x_dot[0])
            #yaw_dot = (x_dot[0] * x_ddot[1] - x_dot[1] * x_ddot[0]) / (np.linalg.norm(x_dot[:2])**2 + 1e-6)
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        
        return flat_output


    def rdp(self,path):
        """
        Ramer-Douglas-Peucker implementation based off pseudocode found at the link below
        https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        """
        eps = 2
        dmax = 0
        idx = 0
        end = len(path)-1
        for i in range(2, end-1):
            d =(path[end] - path[0]).T @  path[i]
            
            if d > dmax:
                idx = i
                dmax = d

        resultPath = []

        if (dmax>eps):
            
            recResults1 = self.rdp(path[0:idx])
            recResults2 = self.rdp(path[idx:end])

            resultPath = np.concatenate((recResults1[:-1],recResults2))

        else:
            resultPath = [path[0],path[end]]

        return resultPath
    
    def simplify_path(self, path, angle_threshold=np.deg2rad(5)):
        """
        Removes points that are nearly collinear based on angle threshold.
        """
        if len(path) < 3:
            return path

        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            prev_vec = path[i] - path[i - 1]
            next_vec = path[i + 1] - path[i]
            prev_vec /= np.linalg.norm(prev_vec)
            next_vec /= np.linalg.norm(next_vec)

            angle = np.arccos(np.clip(np.dot(prev_vec, next_vec), -1.0, 1.0))
            if angle > angle_threshold:
                simplified.append(path[i])
        simplified = np.array(simplified)

        return np.array(simplified)

if __name__ == '__main__':
    # Choose a test example file. You should write your own example files too!
    filename = '../util/test_window.json'
    path = np.load('path.npz')['path']
    traj = WorldTraj(0,0,0,path)

    t= 0

    while t < 10:
        print(traj.update(t))
        t+=0.1