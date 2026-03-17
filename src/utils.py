import yaml

C0 = 299792458.0
KB = 1.38064852e-23

def load_config(path="configs/simulation_parameters.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

"""
Implement polt functions:
    - color map for peridogram
    - target detection
"""
