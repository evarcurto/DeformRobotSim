import pybullet as p
import numpy as np
import time
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math

class DefEnv:
   
    def __init__(self, bullet_client, gui = True) -> None:
        self.bullet_client = bullet_client
        self.gui = gui
        if self.gui == True:
            # connect bullet
            self.bullet_client.connect(self.bullet_client.GUI) #or p.GUI (for test) or p.DIRECT (for train) for non-graphical version
        else:
            self.bullet_client.connect(self.bullet_client.DIRECT) 

        #p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bullet_client.setAdditionalSearchPath(os.path.dirname(__file__) + '/objects') #só posso usar uma vez este comando senão o segundo sobrepom-se ao primeiro

        #directory to save RGBD captures
        self.disk_dir = Path("data/frames/")
        self.disk_dir.mkdir(parents=True, exist_ok=True)

        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.t = 0
        self.state_t = 0
        self.cur_state = 0
        # set gravity
        self.bullet_client.setGravity(0, 0, -9.81)

    def initial_reset(self):
        # reset pybullet to deformable object
        self.bullet_client.resetSimulation(self.bullet_client.RESET_USE_DEFORMABLE_WORLD)
        
        self.load_plane()
        self.load_table()
        self.load_deformable_object()
        self.load_cube() 

        # robot initialization
        self.load_panda()
        self.panda_initial_positions()

        self.camera_system(4,True)
        
    def load_plane(self):
        self.planeId = self.bullet_client.loadURDF("plane/plane.urdf", useFixedBase=True,flags=self.flags)

    def load_table(self):
        self.table_startPos = [0, 0, 0]
        self.tableId= self.bullet_client.loadURDF("table/table.urdf", self.table_startPos, useFixedBase=True,flags=self.flags)
        self.table_height = 0.625
	
    def load_deformable_object(self):
        self.def_startPos = [0.35+0.06, 0, self.table_startPos[2]+self.table_height+0.05] #caso cilindro na horizontal
        self.def_startOrientation = self.bullet_client.getQuaternionFromEuler([0,1.57,0]) #caso cilindro na horizontal
        #self.def_startPos = [0.25, 0, self.table_startPos[2]+self.table_height+0.5] #caso cilindro na vertical
        #self.def_startOrientation = self.bullet_client.getQuaternionFromEuler([0,0,0]) #caso cilindro na vertical
        
        #self.defId = self.bullet_client.loadSoftBody(fileName= 'Mymeshes/hollow_cylinder_scaled.vtk',basePosition = self.def_startPos, 
        #baseOrientation=self.def_startOrientation, scale = 0.8, mass = 0.2, useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1,collisionMargin = 0.001)
        
        self.defId = self.bullet_client.loadSoftBody(fileName= 'Mymeshes/hollow_cylinder_scaled.vtk',basePosition = self.def_startPos, 
        baseOrientation=self.def_startOrientation, scale = 0.8,  mass = 0.1, useNeoHookean = 1, collisionMargin = 0.001, frictionCoeff = 0.5,
                        NeoHookeanMu = 15, NeoHookeanLambda = 10, NeoHookeanDamping = 0.005)



    def load_cube(self):
        self.cube_startPos = [0.7, 0, self.table_startPos[2]+self.table_height+0.05]
        self.cubeId = self.bullet_client.loadURDF("cube/cube.urdf", self.cube_startPos, useFixedBase=True, globalScaling = 0.1,flags=self.flags)

    def load_panda(self):
        self.panda_startPos = [-0.6, 0, self.table_startPos[2]+self.table_height]
        panda_model = "franka_panda/panda.urdf"
        self.pandaId = self.bullet_client.loadURDF(panda_model, basePosition=self.panda_startPos, useFixedBase=True,flags=self.flags)
        self.panda_num_joints = self.bullet_client.getNumJoints(self.pandaId) # 12joints(caso franka panda)
        print("Panda num joints: ", self.panda_num_joints)

        self.dof = self.bullet_client.getNumJoints(self.pandaId)-1
        print("Panda DOF: ", self.dof)
        self.joints = range(self.dof)	

    def panda_initial_positions(self):
        self.useNullSpace = 1
        self.ikSolver = 0
        self.pandaEndEffectorIndex = 11 #8
        self.pandaNumDofs = 7
        self.ll = [-7]*self.pandaNumDofs
        #upper limits for null space (todo: set them to proper range)
        self.ul = [7]*self.pandaNumDofs
        #joint ranges for null space (todo: set them to proper range)
        self.jr = [7]*self.pandaNumDofs
        #restposes for null space
        self.jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        self.rp = self.jointPositions

        index = 0
 
        for j in range(self.panda_num_joints):
            self.bullet_client.changeDynamics(self.pandaId, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.pandaId, j)
            #print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.pandaId, j, self.jointPositions[index]) 
                index=index+1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.pandaId, j, self.jointPositions[index]) 
                index=index+1
        
    def grab_object(self):
        self.camera_system(4,False)
        print("self.state=",self.state)
        print("self.state_t=", self.state_t)
        if self.state==4:
            self.finger_target = 0.08
            self.create_anchors()
        if self.state==2:
            self.finger_target = 0.1
        self.bullet_client.submitProfileTiming("step")
        
        self.update_state()

        if self.state==1 or self.state==3 or self.state==5 or self.state==6:
            self.gripper_height = self.table_startPos[2]+self.table_height+0.053
            if self.state==1 or self.state==5 or self.state==6:
                self.gripper_height = self.table_startPos[2]+self.table_height+0.25

            t = self.t
            self.t += self.control_dt
            if self.state == 1 or self.state== 3 or self.state == 5:
                pos, o = self.bullet_client.getBasePositionAndOrientation(self.defId)
                pos = [pos[0]-0.2, pos[1], self.gripper_height]
                self.prev_pos = pos
            if self.state == 6:
                pos = self.prev_pos
                diffX = pos[0] 
                diffY = pos[1] 
                self.prev_pos = [self.prev_pos[0] - diffX*0.1, self.prev_pos[1]-diffY*0.1, self.prev_pos[2]]
        
            orn = self.bullet_client.getQuaternionFromEuler([math.pi,0.,0.])
            self.bullet_client.submitProfileTiming("IK")
            jointPoses = self.bullet_client.calculateInverseKinematics(self.pandaId,self.pandaEndEffectorIndex, pos, orn, self.ll, self.ul,
            self.jr, self.rp, maxNumIterations=20)
            self.bullet_client.submitProfileTiming()
            for i in range(self.pandaNumDofs):
                self.bullet_client.setJointMotorControl2(self.pandaId, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            self.bullet_client.setJointMotorControl2(self.pandaId, i, self.bullet_client.POSITION_CONTROL,self.finger_target ,force= 10)

    def create_anchors(self):
        data = p.getMeshData(self.defId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        #For the case of the hollow cylinder
        for i in range(1,106,2):
            pose =list(data[1][i])
            p.createSoftBodyAnchor(self.defId ,i, self.cubeId, -1) #anchor: the right face of the cylinder is fixed to the cube
            #p.addUserDebugText("*", pose, textColorRGB=[0,0,0])
        for i in range(0,106,2):
            p.createSoftBodyAnchor(self.defId,i, self.pandaId, self.pandaEndEffectorIndex)

    def camera_system(self, num_cameras: int, show_views: bool):
        self.width = 640
        self.height = 480
        
        fov = 70
        aspect = self.width / self.height
        near = 0.02
        far = 3

        cameraTargetPosition=[0.0,0.0,0.0]
        distance=2
        #yaw=[0, 90, 180, 270]
        yaw=[0+30, 90+30, 180+30, 270+30]
        pitch=-50
        roll=0
        upAxisIndex=2

        self.rgba_array = []
        self.depth_opengl_array =[]
        for i in range(num_cameras):
            #view_matrix = p.computeViewMatrix([0, 0, 2.5], [0, 0, 0], [1, 0, 0])
            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition,distance,yaw[i],pitch,roll,upAxisIndex)
            projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)      

            width, height, rgbImg, depthImg, segImg = self.camera_rend(view_matrix,projection_matrix)  

            rgbBuffer = np.reshape(rgbImg, (height, width, 4))
            rgbArr = np.asarray(rgbBuffer)
            self.rgba_array.append(rgbArr)

            depth_buffer_opengl = np.reshape(depthImg, [height, width])
            far_ = 1
            depth_opengl = far_ * near / (far_ - (far_ - near) * depth_buffer_opengl)
            self.depth_opengl_array.append(depth_opengl)

            if show_views==True:
                plt.subplot(2,2,i+1)
                plt.imshow(depth_opengl, cmap='gray', vmin=0, vmax=1)
                plt.title('Depth cam %i' %(i+1))
        plt.show()
        #print(np.array(self.depth_opengl_array).shape)
                   
        p.stepSimulation()
        
    def camera_rend(self, view_matrix, projection_matrix):

        width, height, rgbImg, depthImg, segImg  = p.getCameraImage(self.width, self.height, view_matrix, projection_matrix)
        return width, height, rgbImg, depthImg, segImg

    def RGBDcapture(self,num_cameras, capture: bool, i: float):
        if capture == True:
            for c in range(num_cameras):
                image_id=i
                print("image_id=", image_id)
                Image.fromarray(self.rgba_array[c]).save(self.disk_dir / f"rgb_cam{c+1}_im{image_id}.png")
                new_p1 = Image.fromarray((self.depth_opengl_array[c]* 255).astype(np.uint8))
                new_p1.save(self.disk_dir / f"depth_cam{c+1}_im{image_id}.png")
            
    def show_cartesian_sliders(self):
        self.list_slider_cartesian = []
        self.list_slider_cartesian.append(p.addUserDebugParameter("VX", -1, 1, 0))
        self.list_slider_cartesian.append(p.addUserDebugParameter("VY", -1, 1, 0))
        self.list_slider_cartesian.append(p.addUserDebugParameter("VZ", -1, 1, 0))
        self.list_slider_cartesian.append(p.addUserDebugParameter("Theta_Dot", -1, 1, 0))    

    def apply_cartesian_sliders(self):
        action = np.empty(4, dtype=np.float64)
        
        for i in range(4):
            action[i] = p.readUserDebugParameter(self.list_slider_cartesian[i])
        
        self.set_action(action)
        p.stepSimulation()

    def set_action(self, action):
        cur_state = p.getLinkState(self.pandaId, self.panda_end_eff_idx)
        cur_pos = np.array(cur_state[0])
        cur_orien = np.array(cur_state[1])
        
        new_pos = cur_pos + np.array(action[:3]) * self.max_vel * self.dt
        #new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high) #to limit space
        
        jointPoses = p.calculateInverseKinematics(self.pandaId, self.panda_end_eff_idx, new_pos, cur_orien)[0:7]
        
        for i in range(len(jointPoses)):
            p.setJointMotorControl2(self.pandaId, i, p.POSITION_CONTROL, jointPoses[i],force=10 * 240.)

class PandaSimAuto(DefEnv):
  def __init__(self, bullet_client):
    DefEnv.__init__(self, bullet_client)
    self.state_t = 0
    self.cur_state = 0
    #self.states=[0,1,2,3,4,5,6]
    self.states=[0,1,2,3,4,5,6]
    self.state_durations=[0.01,0.2,0.1,0.05,0.05,0.2,0.5]
  
  def update_state(self):
    self.state_t += self.control_dt
    if self.state==5 or self.state==6:
        inst_frame = round(self.state_t,3)
        self.RGBDcapture(4, False, inst_frame)
    if self.state_t > self.state_durations[self.cur_state]:
      self.cur_state += 1
      if self.cur_state>=len(self.states):
        self.cur_state = 5
      self.state_t = 0
      self.state=self.states[self.cur_state]
      print("self.state=",self.state)