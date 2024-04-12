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
import torchvision
import paho.mqtt.client as mqtt
from torchvision.models import resnet50
import sys
import cv2 as cv
import pytesseract
import re
import traceback
import os                                                             
                                                                      
                                                               
def get_video_name(video_path):                                       
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
    try:
        layerNames = net.getLayerNames()
        return [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    except Exception as e:
        print("Erreur lors de la récupération des noms de sortie:", e)

class VehicleClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VehicleClassifier, self).__init__()
        self.resnet50 = resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)
    
def preprocess_vehicle_region(vehicle_region):
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
        print("Erreur lors du prétraitement de la région du véhicule:", e)

class_names = {0: "France", 1: "Espagne"}

try:
    model = VehicleClassifier(num_classes=2)
    model.load_state_dict(torch.load("Nationality.pth"))
    model.eval()
except Exception as e:
    print(f"Erreur lors du chargement du modèle de classification de nationalité : {e}")
    sys.exit()

def matricule(frame, outs, width_factor=1.1, height_factor=1.0):
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
                c = class_names.get(predicted_class, "Inconnu")
    
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(plate, config=custom_config)
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text)
        
            if cleaned_text:
                return cleaned_text, c

        return "Non detecte", "Inconnu"
    except Exception as e:
        print("Erreur lors de la détection de la plaque d'immatriculation:", e)

class VehicleCounter:
    def __init__(self, video_path):
        try:
            self.video_name = get_video_name(video_path) 
            self.broker_address = "127.0.0.1"
            self.broker_port = 1883
            self.topic = "vehicle_data"
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
            print("Erreur lors de la lecture du fichier de classes:", e)
    def send_video_name(self):
        try:
            video_name_json = json.dumps({"video_name": self.video_name})  
            self.publish_json_to_mqtt(video_name_json)  
        except Exception as e:
            print("Erreur lors de l'envoi du nom de la vidéo:", e)
    def publish_video_end_message(self):
        try:
           video_end_message = json.dumps({"video_status": "finished"})
           self.publish_json_to_mqtt(video_end_message)
        except Exception as e:
           print("Erreur lors de l'envoi du message de fin de vidéo :", e)

    def publish_json_to_mqtt(self, json_data):
        try:
            self.mqtt_client.connect(self.broker_address, self.broker_port, 60)
            self.mqtt_client.publish(self.topic, json_data, qos=0)
            self.mqtt_client.disconnect()
        except Exception as e:
            print("Erreur lors de l'initialisation:", e)
            traceback.print_exc()
            
    def process_video(self):
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
                         json_data = None  # Initialisation de json_data à None
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
                             if c != "Inconnu" and m != "Non detecte":
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
          print(f"Erreur lors du traitement de la vidéo : {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    vc = VehicleCounter(video_path)
    vc.process_video()
