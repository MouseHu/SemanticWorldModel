from gridworld.fourrooms_water import *
import time
if __name__=="__main__":
    #check whether waters and coins are fixed.
    env = ImageInputWarpper(FourroomsWaterNorender())
    time.sleep(2)
    env.reset()
    env.reset()