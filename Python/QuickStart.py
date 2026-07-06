import os
os.chdir(r"C:\Users\azmib\Desktop\Python")
from BosonSDK import CamAPI

myCam = CamAPI.pyClient(manualport='COM3')  # replace COM7 with your actual port
myCam.bosonRunFFC()
result, serialnumber = myCam.bosonGetCameraSN()
print(serialnumber)
myCam.Close()