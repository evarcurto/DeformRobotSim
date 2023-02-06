import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import franka_emika_panda_pybullet
from franka_emika_panda_pybullet.panda_robot import PandaRobot
from franka_emika_panda_pybullet.movement_datasets import read_fep_dataset
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt



class DefEnv:
   
    def __init__(self, gui = True) -> None:

        self.gui = gui
        if self.gui == True:
            # connect bullet
            p.connect(p.GUI) #or p.GUI (for test) or p.DIRECT (for train) for non-graphical version
        else:
            p.connect(p.DIRECT) 

        #p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(os.path.dirname(__file__) + '/objects')
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.include_gripper = True

        self.timeStep = 0.003
        self.n_substeps = 20
        self.dt = self.timeStep*self.n_substeps
        self.max_vel = 100

        #directory to save RGBD captures
        self.disk_dir = Path("data/frames/")
        self.disk_dir.mkdir(parents=True, exist_ok=True)

        

    def initial_reset(self):
        # reset pybullet to deformable object
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        # simulation time
        p.setGravity(0, 0, -9.81)

        self.load_plane()
        self.load_table()
        self.load_deformable_object()
        self.load_cube() 
        self.create_anchors()
        self.load_panda()

        self.camera_system(4,True)
        p.stepSimulation()

        #self.panda_trajectory()
        self.my_panda_trajectory()
        p.stepSimulation()

        #p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)
        #p.setTimeStep(self.timeStep)
        p.stepSimulation()

        
    def load_plane(self):
        self.planeId = p.loadURDF("plane/plane.urdf", useFixedBase=True)

    def load_table(self):
        #self.table_startPos = [0, 0, 0.81]
        self.table_startPos = [0, 0, 0]
        self.tableId= p.loadURDF("table/table.urdf", self.table_startPos, useFixedBase=True)
        self.table_height = 0.625
	
    def load_deformable_object(self):
        #self.def_startPos = [panda_eff_state[0][0], panda_eff_state[0][1], 0.0]
        self.def_startPos = [0.35, 0, self.table_startPos[2]+self.table_height+0.05] #caso cilindro na horizontal
        self.def_startOrientation = p.getQuaternionFromEuler([0,1.57,0]) #caso cilindro na horizontal
        #self.def_startPos = [0.25, 0, self.table_startPos[2]+self.table_height+0.5] #caso cilindro na vertical
        #self.def_startOrientation = p.getQuaternionFromEuler([0,0,0]) #caso cilindro na vertical
        
        self.defId = p.loadSoftBody(fileName= 'Mymeshes/hollow_cylinder_scaled.vtk',basePosition = self.def_startPos, baseOrientation=self.def_startOrientation, scale = 1, mass = 1, useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1,collisionMargin = 0.001)

    def load_cube(self):
        self.cube_startPos = [0.7, 0, self.table_startPos[2]+self.table_height+0.05]
        self.cubeId = p.loadURDF("cube/cube.urdf", self.cube_startPos, useFixedBase=True, globalScaling = 0.1)

    def load_panda(self):
        self.panda_startPos = [-0.6, 0, self.table_startPos[2]+self.table_height]
        self.panda_startOrientation = self.startOrientation
        panda_model = "model_description/panda_with_gripper.urdf" if self.include_gripper else "model_description/panda.urdf"
        self.pandaId = p.loadURDF(panda_model, basePosition=self.panda_startPos, baseOrientation=self.panda_startOrientation, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        self.panda_num_joints = p.getNumJoints(self.pandaId) # 12joints(caso franka panda)	7-joint franka emika panda without gripper 10-joint franka emika panda with gripper
        print("Panda num joints: ", self.panda_num_joints)

        self.dof = p.getNumJoints(self.pandaId)-1
        print("Panda DOF: ", self.dof)
        self.joints = range(self.dof)	

    def create_anchors(self):
        data = p.getMeshData(self.defId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        #For the case of the hollow cylinder
        for i in range(1,106,2):
            pose =list(data[1][i])
            p.createSoftBodyAnchor(self.defId ,i, self.cubeId, -1) #anchor: the right face of the cylinder is fixed to the cube
            #p.addUserDebugText("*", pose, textColorRGB=[0,0,0])
        #for i in range(0,106,2):
        #    p.createSoftBodyAnchor(self.defId,i, self.pandaId, self.panda_end_eff_idx)

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
  
    def my_panda_trajectory(self):
        # Set up variables for simulation loop
        SAMPLING_RATE = 1e-2  # 1000Hz sampling rate
        period = 1 / SAMPLING_RATE
        self.counter_seconds = -1

        pos = np.array([[0, 0, 0.65],
                   [0.1, 0, 0.60],
                   [0.2, 0, 0.55],
                   [0.3, 0, 0.50],
                   [0.3, 0, 0.45],
                   [0.35, 0, 0.40],
                   [0.40, 0, 0.35],
                   [0.45, 0, 0.00]])
        dataset_length = pos.shape[0]

        # start simulation loop
        for i in range(dataset_length):
        # Print status update every second of the simulation
            if i % period == 0:
                self.counter_seconds += 1
                print("Passed time in simulation: {:>4} sec".format(self.counter_seconds))

            #cur_state = p.getLinkState(self.pandaId, self.dof)
            #cur_pos = np.array(cur_state[0])
            #new_pos = cur_pos + pos[i]

            JointPos = self.calculate_inverse_kinematics(pos[i])
            #print(JointPos) #9
            #converter sel.JointPos em lista antes de usar set_target_positions
            self.set_target_positions(JointPos)

            # Perform simulation step
            p.stepSimulation()
            time.sleep(SAMPLING_RATE)

    def calculate_inverse_kinematics(self, position):
        #Returns a list of joint positions for each degree of freedom, so the legth of this list is the number of degrees of freedom
        Jointpos = p.calculateInverseKinematics(self.pandaId, self.dof, position, solver=0)
        return list(Jointpos)

    def set_target_positions(self, desired_Jointpos):
        # If robot set up with gripper, set those positions to 0
        p.setJointMotorControlArray(bodyUniqueId=self.pandaId,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=desired_Jointpos)
    
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

    def RGBDcapture(self,num_cameras, capture: bool, i: int):
        if capture == True:
            for c in range(num_cameras):
                image_id=i
                Image.fromarray(self.rgba_array[c]).save(self.disk_dir / f"rgb_cam{c+1}_im{image_id}.png")
                new_p1 = Image.fromarray((self.depth_opengl_array[c]* 255).astype(np.uint8))
                new_p1.save(self.disk_dir / f"depth_cam{c+1}_im{image_id}.png")
            

        


   