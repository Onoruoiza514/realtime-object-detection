from roboflow import Roboflow

rf = Roboflow(api_key="DPHfXgNOKyDrQM7YchCC")
project = rf.workspace("david-lee-d0rhs").project("american-sign-language-letters")
version = project.version(6)
dataset = version.download("yolov8", location="data/asl_dataset")
dataset