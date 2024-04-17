Cette couche prend en charge l'intégration de l'application de détection et de classification des véhicules à partir d'une vidéo, avec l'envoi de messages MQTT contenant un JSON pour chaque véhicule détecté dans une image Yocto, dédié aux cartes embarquées utilisant l'architecture x86.

Pour le bon fonctionnement de cette application, il est nécessaire d'ajouter d'autres couches pour inclure les bibliothèques nécessaires ainsi que d'autres technologies utiles :

   - **meta-openembedded**
   - **meta-vehicle** qui contient des recettes spécifiques à l'apprentissage automatique (recipes-ml) essentielles pour l'intégration des bibliothèques telles que PyTorch, TorchVision et PyTesseract.
