import pybullet as p
import time
from deform_env import DefEnv
from deform_env import PandaSimAuto


def main():

    panda = PandaSimAuto(p)
    timeStep=1./240.
    #timeStep=1/120.
    p.setTimeStep(timeStep)
    panda.control_dt = timeStep
    panda.initial_reset()
    
    for i in range (100000):
        panda.grab_object()
        p.stepSimulation()
        time.sleep(timeStep)

   
if __name__ == '__main__':
    main()

