'''Taille d'entrée des images pour le modèle YOLO'''
INPUT_SIZE = 320

''''' Seuil de confiance pour la détection d'objets'''
CONFIDENCE_THRESHOLD = 0.27
'''Seuil de suppression non maximale (NMS) pour supprimer les détections redondantes'''
NMS_THRESHOLD = 0.2

''' Couleur de la police pour l'affichage des étiquettes'''
FONT_COLOR = (0, 0, 255)
''' Taille de la police pour l'affichage des étiquettes'''
FONT_SIZE = 0.5
''' Épaisseur de la police pour l'affichage des étiquettes'''
FONT_THICKNESS = 1

''' Chemin du fichier contenant les noms des classes du jeu de données COCO'''
CLASSES_FILE = "./coco.names"
''' Indices des classes requises pour la détection (par exemple: personnes, voitures, motos, camions)'''
REQUIRED_CLASS_INDEX = [2, 3, 5, 7]

'''Chemin de configuration du modèle YOLO'''
MODEL_CONFIG = './yolov4.cfg'
''' Chemin des poids pré-entraînés du modèle YOLO'''
MODEL_WEIGHTS = './yolov4.weights'
