import json
import time
import cv2
import numpy as np
import config
from utils import EuclideanDistTracker, postProcess
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
import paho.mqtt.client as mqtt
from torchvision.models import resnet50
import sys
import cv2 as cv
import pytesseract
import re
import traceback
import os                                                                                                                                  
import configparser


"""Lecture de fihcier de configuration de broker"""
broker_cfg = configparser.ConfigParser()
broker_cfg.read('broker.cfg')

"""Adresse et port du broker MQTT"""
broker_address = broker_cfg.get('Broker', 'broker_address')
broker_port = broker_cfg.getint('Broker', 'broker_port')

""" Sujet MQTT sur lequel écouter les données des véhicules"""
topic = broker_cfg.get('Broker', 'topic')   
                                                                  
def get_video_name(video_path):
    """
    Obtient le nom de la vidéo à partir du chemin du fichier.
    
    Args:
        video_path (str): Le chemin du fichier vidéo.
        
    Returns:
        str: Le nom de la vidéo.
    """        
    return os.path.basename(video_path) 
    
confThreshold = 0.5  
nmsThreshold = 0.4  
inpWidth = 416  
inpHeight = 416  

classesFile = "classes.names"
classes = None
try:
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
except FileNotFoundError as e:
    print(f"Error: Classes file '{classesFile}' not found: {e}")
    sys.exit(1)

modelConfiguration = "./darknet-yolov3.cfg"
modelWeights = "./model.weights"

try:
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
except cv.error as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def getOutputsNames(net):
    """
    Obtient les noms des couches de sortie du réseau.
    
    Args:
        net: Le réseau de neurones.
        
    Returns:
        List[str]: Liste des noms des couches de sortie.
    """
    try:
        layerNames = net.getLayerNames()
        return [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    except Exception as e:
        print("Error retrieving output names:", e)
class ModelLoader:
    """
    Classe pour charger les modèles une fois et les réutiliser dans le code.
    """
    def __init__(self, num_classes=2):
        self.resnet_model = resnet50(pretrained=True)
        in_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(in_features, num_classes)

class VehicleClassifier(nn.Module):
    """
    Classe pour le classificateur de véhicules.
    """
    def __init__(self, model_loader):
        super(VehicleClassifier, self).__init__()
        self.resnet50 = model_loader.resnet_model

    def forward(self, x):
        return self.resnet50(x)
    
def preprocess_vehicle_region(vehicle_region):
    """
    Prétraite la région du véhicule pour la classification.
    
    Args:
        vehicle_region: La région du véhicule à prétraiter.
        
    Returns:
        torch.Tensor: Image prétraitée.
    """
    try:
        if not isinstance(vehicle_region, Image.Image):
            vehicle_region = Image.fromarray(vehicle_region)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = transform(vehicle_region)
        image = image.unsqueeze(0)
        return image
    except Exception as e:
        print("Error during vehicle region preprocessing:", e)

class_names = {0: "France", 1: "Espagne"}

try:
    model_loader = ModelLoader(num_classes=2) 
    model = VehicleClassifier(model_loader)  
    model.load_state_dict(torch.load("Nationality.pth"))
    model.eval()
except Exception as e:
    print(f"Error loading nationality classification model: {e}")
    sys.exit()

def matricule(frame, outs, width_factor=1.1, height_factor=1.0):
    """
    Détecte et extrait la plaque d'immatriculation du véhicule dans le frame.
    
    Args:
        frame: Le frame d'entrée.
        outs: Sorties du modèle YOLO.
        width_factor (float): Facteur d'agrandissement de la largeur de la boîte de détection.
        height_factor (float): Facteur d'agrandissement de la hauteur de la boîte de détection.
        
    Returns:
        str: Le numéro de la plaque d'immatriculation détecté.
        str: Le pays associé à la plaque d'immatriculation.
    """
    try:
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        detected_plates = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]

                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2 * width_factor)
                    top = int(center_y - height / 2 * height_factor)
                    width = int(width * width_factor)
                    height = int(height * height_factor)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    detected_plates.append(frame[top:top+height, left:left+width])

        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        detected_plates = []

        for i in range(len(indices)):
            idx = indices[i][0] if isinstance(indices[i], list) else indices[i]
            box = boxes[idx]
            left, top, width, height = box[0], box[1], box[2], box[3]
            detected_plates.append(frame[top:top+height, left:left+width])

        for plate in detected_plates:
            with torch.no_grad():
                input_plate = cv2.imwrite("plate_temp.jpg", plate)
                input_plate = cv2.imread("plate_temp.jpg")
                input_tensor = preprocess_vehicle_region(input_plate)
                model.eval()
                predictions = model(input_tensor)
                predicted_class = torch.argmax(predictions).item()
                c = class_names.get(predicted_class, "Unknown")
            if os.path.exists("plate_temp.jpg"):
                os.remove("plate_temp.jpg")

            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(plate, config=custom_config)
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text)
        
            if cleaned_text:
                return cleaned_text, c

        return "Not detected", "Unknown"
    except Exception as e:
        print("Error during license plate detection:", e)

class VehicleCounter:
    """
    Classe pour le compteur de véhicules.
    """
    def __init__(self, video_path):
        try:
            self.video_name = get_video_name(video_path) 
            self.broker_address = broker_address 
            self.broker_port = broker_port
            self.topic = topic
            self.mqtt_client = mqtt.Client()
            self.tracker = EuclideanDistTracker()
            self.cam = cv2.VideoCapture(video_path)
            self.input_size = config.INPUT_SIZE
            self.confThreshold = config.CONFIDENCE_THRESHOLD
            self.nmsThreshold = config.NMS_THRESHOLD
            self.classNames = open(config.CLASSES_FILE).read().strip().split('\n')
            self.required_class_index = config.REQUIRED_CLASS_INDEX
            modelConfiguration = config.MODEL_CONFIG
            modelWeights = config.MODEL_WEIGHTS
            self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
            np.random.seed(42)
            self.colors = np.random.randint(0, 255, size=(len(self.classNames), 3), dtype='uint8')
        except Exception as e:
            print("Error reading class file:", e)
            
    def cleanup(self):
        """
        Nettoie les ressources utilisées, comme les captures vidéo OpenCV.
        """
        try:
            self.cam.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print("Error during cleanup:", e)
            
    def send_video_name(self):
        """
        Envoie le nom de la vidéo au serveur MQTT.
        """
        try:
            video_name_json = json.dumps({"video_name": self.video_name})  
            self.publish_json_to_mqtt(video_name_json)  
        except Exception as e:
            print("Error sending video name:", e)
    def publish_video_end_message(self):
        """
        Envoie le message de fin de la vidéo au serveur MQTT.
        """
        try:
           video_end_message = json.dumps({"video_status": "finished"})
           self.publish_json_to_mqtt(video_end_message)
        except Exception as e:
           print("Error sending end of video message :", e)

    def publish_json_to_mqtt(self, json_data):
        """
        Envoie les données JSON au serveur MQTT.
        
        Args:
            json_data: Les données JSON à envoyer.
        """
        try:
            self.mqtt_client.connect(self.broker_address, self.broker_port, 60)
            self.mqtt_client.publish(self.topic, json_data, qos=0, retain=True)
            self.mqtt_client.disconnect()
        except Exception as e:
            print("Error during initialization:", e)
            traceback.print_exc()
            
    def process_video(self):
         """
         Traite la vidéo en détectant les véhicules et leurs informations associées.
         """
         try:
             self.send_video_name()
             s = time.time()
             frame_counter = 0  
             analyze_frame = True  
             while self.cam.isOpened():
                 ret, frame = self.cam.read()
                 if ret:
                     if analyze_frame:
                         
                         blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.input_size, self.input_size), [0, 0, 0], 1, crop=False)
                         self.net.setInput(blob)
                         layersNames = self.net.getLayerNames()
                         outputNames = [layersNames[i - 1] for i in self.net.getUnconnectedOutLayers()]
                         outputs = self.net.forward(outputNames)

                         closest_vehicle = postProcess(outputs, frame, self.colors, self.classNames, self.confThreshold, self.nmsThreshold,
                                            self.required_class_index, self.tracker)
                         json_data = None  
                         if closest_vehicle:
                             cv2.imwrite("screenshot.jpg", frame)
                             with torch.no_grad():
                                 blob1 = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
                                 net.setInput(blob1)
                                 outs1 = net.forward(getOutputsNames(net))
                                 m, c = matricule(frame, outs1)
                            
                               
                             json_data = {
                                     "activity": "Monitoring",
                                     "class": closest_vehicle['name'],
                                     "classificators": [{
                                         "make": closest_vehicle['make'],
                                         "model": closest_vehicle['model'],
                                         "class": closest_vehicle['name'],
                                         "color": closest_vehicle['colors'],
                                         "country": c,
                                         "registration": m
                                     }],
                                     "registration": m
                                 }

                             json_output = json.dumps(json_data, indent=4)
                             if c != "Unknown" and m != "Not detected":
                                 print(json_output)
                                 self.publish_json_to_mqtt(json_output)
                                 print("Message sent")
                         analyze_frame = False  
                     else:
                         
                         frame_counter += 1  
                         if frame_counter == 7:  
                             frame_counter = 0
                             analyze_frame = True
                 else:
                     print("time finished !!")
                     self.publish_video_end_message()
                     break
         except Exception as e:
          print(f"Error during video processing : {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    vc = VehicleCounter(video_path)
    try:
       vc.process_video()
    finally:
       vc.cleanup()

