import math
import os
import cv2
import numpy as np
from collections import Counter
import torch
from torchvision import transforms, models
from torch import nn
import json
from PIL import Image

def load_checkpoint(filepath):
    """
    Charge un modèle à partir d'un fichier checkpoint.

    Args:
        filepath (str): Chemin vers le fichier checkpoint.

    Returns:
        torch.nn.Module: Modèle chargé.
    """
    try:
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 140)
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.class_to_idx = checkpoint['class_to_idx']
        return model
    except Exception as e:
        print("Erreur lors du chargement du modèle :", e)

brand = load_checkpoint('../src/brand.pth')
if brand:
    brand.eval()

try:
    with open('../src/marque.json', 'r') as f:
        class_to_idx = json.load(f)
except FileNotFoundError as e:
    print("Erreur lors de l'ouverture du fichier JSON :", e)
else:
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

img_transforms = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def tansf_image(image):
    """
    Transforme une image en un format utilisable par le modèle.

    Args:
        image (numpy.ndarray): Image d'entrée.

    Returns:
        torch.Tensor: Image transformée.
    """
    try:
        image_pil = Image.fromarray(image)
        image_transformed = img_transforms(image_pil).float()
        image_transformed = image_transformed.unsqueeze(0)
        return image_transformed
    except Exception as e:
        print("Erreur lors de la transformation de l'image :", e)

class EuclideanDistTracker:

    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        """
        Met à jour le suivi des objets avec de nouveaux rectangles.

        Args:
            objects_rect (list): Liste des rectangles d'objets détectés.

        Returns:
            list: Liste des rectangles mis à jour avec les ID des objets.
        """
        try:
            objects_bbs_ids = []
            for rect in objects_rect:
                x, y, w, h, index = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                same_object_detected = False
                for id, pt in self.center_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])
                    if dist < 25:
                        self.center_points[id] = (cx, cy)
                        objects_bbs_ids.append([x, y, w, h, id, index])
                        same_object_detected = True
                        break
                if same_object_detected is False:
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                    self.id_count += 1
            new_center_points = {}
            for obj_bb_id in objects_bbs_ids:
                _, _, _, _, object_id, index = obj_bb_id
                center = self.center_points[object_id]
                new_center_points[object_id] = center
            self.center_points = new_center_points.copy()
            return objects_bbs_ids
        except Exception as e:
            print("Erreur lors de la mise à jour du suivi des objets :", e)

def get_color_name(hsv_value):
    """
    Obtient le nom de la couleur à partir de sa valeur HSV.

    Args:
        hsv_value (tuple): Tuple contenant les valeurs HSV.

    Returns:
        str: Nom de la couleur.
    """
    try:
        h, s, v = hsv_value

        if v < 50:
            return "Noir"
        if s < 50:
            return "Blanc"

        color_ranges = [
            (0, 15, "Rouge"),
            (15, 45, "Jaune"),
            (45, 90, "Vert"),
            (90, 120, "Cyan"),
            (120, 150, "Bleu"),
            (150, 165, "Magenta"),
            (0, 10, "Orange"),
            (10, 20, "Gris"),
        ]

        for start, end, color in color_ranges:
            if (start <= h < end) or (165 <= h <= 180 and start == 0):
                return color

        return "Inconnu"
    except Exception as e:
        print("Erreur lors de la détermination de la couleur :", e)

def get_color(vehicle_region, v1_min, v2_min, v3_min, v1_max, v2_max, v3_max):
    """
    Obtient la couleur dominante d'une région de véhicule.

    Args:
        vehicle_region (numpy.ndarray): Région de l'image contenant le véhicule.
        v1_min (int): Valeur minimale de la première composante HSV.
        v2_min (int): Valeur minimale de la deuxième composante HSV.
        v3_min (int): Valeur minimale de la troisième composante HSV.
        v1_max (int): Valeur maximale de la première composante HSV.
        v2_max (int): Valeur maximale de la deuxième composante HSV.
        v3_max (int): Valeur maximale de la troisième composante HSV.

    Returns:
        str: Nom de la couleur dominante.
    """
    try:
        if vehicle_region is None or vehicle_region.size == 0:
            return None

        vehicle_hsv = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2HSV)

        if vehicle_hsv is None or vehicle_hsv.size == 0:
            return None

        mask = cv2.inRange(vehicle_hsv, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            hsv_value = vehicle_hsv[cy, cx]
            color_name = get_color_name(hsv_value)

            return color_name

        return None
    except Exception as e:
        print("Erreur lors de la détermination de la couleur du véhicule :", e)

def dist_calculator(box_width, img_w):
    """
    Calcule la distance entre la caméra et un objet à partir de la largeur du cadre.

    Args:
        box_width (int): Largeur du cadre contenant l'objet.
        img_w (int): Largeur de l'image.

    Returns:
        float: Distance entre la caméra et l'objet.
    """
    try:
        focal_length = 1000
        known_width = 100
        distance = (known_width * focal_length) / box_width
        return distance
    except Exception as e:
        print("Erreur lors du calcul de la distance :", e)

def find_closest_vehicle(detected_vehicles, img_width):
    """
    Trouve le véhicule le plus proche parmi une liste de véhicules détectés.

    Args:
        detected_vehicles (list): Liste des véhicules détectés.
        img_width (int): Largeur de l'image.

    Returns:
        dict: Détails du véhicule le plus proche.
    """
    try:
        if not detected_vehicles:
            return None

        min_distance = float('inf')
        closest_vehicle = None

        for vehicle in detected_vehicles:
            x, y, w, h = vehicle["position"]
            box_width = w
            distance = dist_calculator(box_width, img_width)
            if distance < min_distance:
                min_distance = distance
                closest_vehicle = vehicle

        return closest_vehicle
    except Exception as e:
        print("Erreur lors de la recherche du véhicule le plus proche :", e)

def postProcess(outputs, img, colors, classNames, confThreshold, nmsThreshold, required_class_index, tracker):
    """
    Effectue le post-traitement des résultats de détection d'objet.

    Args:
        outputs (list): Sorties du modèle de détection d'objet.
        img (numpy.ndarray): Image d'entrée.
        colors (list): Liste des couleurs.
        classNames (list): Liste des noms de classe.
        confThreshold (float): Seuil de confiance.
        nmsThreshold (float): Seuil de suppression non maximale.
        required_class_index (list): Liste des index de classe requis.
        tracker (EuclideanDistTracker): Instance du suivi de distance euclidienne.

    Returns:
        dict: Détails du véhicule le plus proche.
    """
    try:
        detected_vehicles = []
        height, width = img.shape[:2]
        boxes = []
        classIds = []
        confidence_scores = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if classId in required_class_index and confidence > confThreshold:
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)

        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detected_vehicles.append({
                "position": (x, y, w, h),
                "classId": classIds[i],
                "confidence": confidence_scores[i]
            })

        closest_vehicle = find_closest_vehicle(detected_vehicles, width)

        if closest_vehicle:
            x, y, w, h = closest_vehicle["position"]
            classId = closest_vehicle["classId"]
            confidence = closest_vehicle["confidence"]
            vehicle_region = img[y:y+h, x:x+w]
            v1_min, v2_min, v3_min = 0, 0, 0
            v1_max, v2_max, v3_max = 255, 255, 255
            vehicle_color = get_color(vehicle_region, v1_min, v2_min, v3_min, v1_max, v2_max, v3_max)
            name = classNames[classId]
            imgtr= tansf_image(vehicle_region)
            with torch.no_grad():
                output = brand(imgtr)
                probabilities = torch.exp(output)
            top_prob, top_class = probabilities.topk(1, dim=1)
            predicted_class_index = top_class.item()
            predicted_class_name = idx_to_class[predicted_class_index]
            make, model = predicted_class_name.split(' ', 1)
            return {
                "name": name,
                "confidence": int(confidence*100),
                "position": (x, y, w, h),
                "colors": vehicle_color,
                "make": make,
                "model": model
            }

        return None
    except Exception as e:
        print("Erreur lors du post-traitement des résultats :", e)
