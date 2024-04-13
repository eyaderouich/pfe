import json
import csv
import paho.mqtt.client as mqtt
import os
import sys
from fuzzywuzzy import fuzz

"""Adresse et port du broker MQTT"""
broker_address = "127.0.0.1"
broker_port = 1883

""" Sujet MQTT sur lequel écouter les données des véhicules"""
topic = "vehicle_data"

""" Dossier où sauvegarder les fichiers CSV"""
csv_folder = "csv_files"

""" Référence au fichier CSV actuel"""
csv_file = None

""" Booléen indiquant si le tableau de bord doit être lancé"""
launch_dashboard = False

def create_csv_file(video_name):
    """
    Crée un fichier CSV pour stocker les données des véhicules.

    Args:
        video_name (str): Nom de la vidéo associée aux données des véhicules.

    Returns:
        str: Chemin d'accès complet du fichier CSV créé.
    """
    global csv_file
    csv_file_path = os.path.join(csv_folder, f"{video_name}.csv")
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Color", "Make", "Model", "Registration", "Country"])
    csv_file = open(csv_file_path, mode='a', newline='')
    if launch_dashboard:
        start_streamlit(video_name)
    return csv_file_path

def get_last_registration(csv_file):
    """
    Récupère la dernière immatriculation enregistrée dans le fichier CSV.

    Args:
        csv_file (str): Chemin d'accès complet du fichier CSV.

    Returns:
        str: Dernière immatriculation enregistrée.
    """
    try:
        with open(csv_file, 'r', newline='') as file:
            last_line = None
            for last_line in csv.reader(file): pass
            if last_line:
                return last_line[4]  
    except Exception as e:
        print(f"Error reading last registration from CSV: {e}")
    return ""

def write_to_csv(data, csv_file, msg):
    """
    Écrit les données des véhicules dans le fichier CSV.

    Args:
        data (dict): Données des véhicules à écrire dans le CSV.
        csv_file (file): Référence au fichier CSV où écrire les données.
        msg (obj): Objet représentant le message MQTT reçu.
    """
    last_registration = get_last_registration(csv_file.name)
    current_registration = data['registration']
    sort_ratio = fuzz.token_sort_ratio(last_registration, current_registration)

    threshold = 45
    if sort_ratio < threshold:
        with open(csv_file.name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([data['class'], data['classificators'][0]['color'], data['classificators'][0]['make'],
                             data['classificators'][0]['model'], current_registration, data['classificators'][0]['country']])
        print(f"Received message on topic '{msg.topic}': {msg.payload.decode()}")  
        print("Data written to CSV successfully") 

def start_streamlit(video_name):
    """
    Lance l'application Streamlit pour afficher le tableau de bord.

    Args:
        video_name (str): Nom de la vidéo pour laquelle le tableau de bord est lancé.
    """
    import subprocess
    subprocess.Popen(["streamlit", "run", "dashboard.py", "--", video_name])

def on_message(client, userdata, msg):
    """
    Fonction de rappel appelée lors de la réception d'un message MQTT.

    Args:
        client (mqtt.Client): Client MQTT qui a reçu le message.
        userdata: Données utilisateur passées à la fonction de rappel.
        msg (mqtt.MQTTMessage): Objet représentant le message MQTT reçu.
    """
    global csv_file
    try:
        data = json.loads(msg.payload.decode())
        if 'video_name' in data:
            video_name = data['video_name']
            if csv_file:
                csv_file.close()
            create_csv_file(video_name)
            print(f"Created CSV file for video: {csv_file.name}")
        elif 'video_status' not in data:
            write_to_csv(data, csv_file, msg) 
        else:
            print(f"Received message on topic '{msg.topic}': {msg.payload.decode()}")
    except Exception as e:
        print("Error processing JSON data:", e)

def on_connect(client, userdata, flags, rc):
    """
    Fonction de rappel appelée lors de la connexion au broker MQTT.

    Args:
        client (mqtt.Client): Client MQTT qui s'est connecté.
        userdata: Données utilisateur passées à la fonction de rappel.
        flags: Drapeaux de connexion.
        rc (int): Code de retour de la connexion.
    """
    if rc == 0:
        print("Connected to broker")
        client.subscribe(topic, qos=0)
    else:
        print("Failed to connect to broker with code", rc)

"""Vérifier les arguments de ligne de commande pour décider de lancer le tableau de bord"""
if "--dashboard" in sys.argv:
    launch_dashboard = True

"""Initialiser le client MQTT et définir les fonctions de rappel"""
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

"""Se connecter au broker MQTT et démarrer la boucle de réception des messages"""
client.connect(broker_address, broker_port, 60)

