import os
os.chdir(r"C:\Users\azmib\Desktop\Python")
from BosonSDK import CamAPI
myCam = CamAPI.pyClient(manualport='COM3')
methods = [m for m in dir(myCam) if 'radiometry' in m.lower() or 'tstable' in m.lower() or 'agc' in m.lower() or 'dvo' in m.lower()]
print(methods)
myCam.Close()