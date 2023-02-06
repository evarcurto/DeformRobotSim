import pybullet as p
from deform_env import DefEnv

def main():

    # PandaEnv().initial_reset()
    DE = DefEnv() #init
    DE.initial_reset() #initial_reset
    num_cameras =4

    i=0
    while 1:
        p.stepSimulation()
        DE.camera_system(num_cameras,False)
        DE.RGBDcapture(num_cameras,True,i)
        i=i+1
    #p.disconnect()
    

if __name__ == '__main__':
    main()

