import pytest
from main import *
import torch
import numpy as np 
import cv2 


def get_video_name(video_path):                    
    return os.path.basename(video_path)


def test_getOutputsNames():

    class FakeNet:
        def getLayerNames(self):
            return ['layer1', 'layer2', 'layer3']

        def getUnconnectedOutLayers(self):
            return [1, 2]

    fake_net = FakeNet()
    assert getOutputsNames(fake_net) == ['layer1', 'layer2']



def test_VehicleClassifier():
    model = VehicleClassifier(num_classes=2)
    assert isinstance(model, torch.nn.Module)


def test_preprocess_vehicle_region():

    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)

    processed_image = preprocess_vehicle_region(image)

    assert isinstance(processed_image, torch.Tensor)
    assert processed_image.shape == (1, 3, 224, 224)


def test_matricule_no_detection():
    """Tests if the function returns 'Non detecte' and 'Inconnu' when no plates are detected."""
    detected_text, detected_class = matricule(np.zeros((480, 640, 3)), [])
    assert detected_text == "Non detecte"
    assert detected_class == "Inconnu"



def test_VehicleCounter():
    vc = VehicleCounter("./demo.mp4")
    assert vc.video_name == "video.mp4"


def test_send_video_name():
    vc = VehicleCounter("./demo.mp4")
    assert vc.send_video_name() == None 


def test_publish_video_end_message():
    vc = VehicleCounter("./demo.mp4")
    assert vc.publish_video_end_message() == None 

def test_publish_json_to_mqtt():
    vc = VehicleCounter("./demo.mp4")
    assert vc.publish_json_to_mqtt({"test": "data"}) == None 

def test_VehicleCounter():
  
    vc = VehicleCounter("demo.mp4")

    assert isinstance(vc, VehicleCounter)

    assert hasattr(vc, "broker_address")
    assert hasattr(vc, "broker_port")
    assert hasattr(vc, "topic")
    assert hasattr(vc, "mqtt_client")
    assert hasattr(vc, "tracker")
    assert hasattr(vc, "cam")
    assert hasattr(vc, "input_size")
    assert hasattr(vc, "confThreshold")
    assert hasattr(vc, "nmsThreshold")
    assert hasattr(vc, "classNames")
    assert hasattr(vc, "required_class_index")
    assert hasattr(vc, "net")
    assert hasattr(vc, "colors")

    