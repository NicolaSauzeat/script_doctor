# script_doctor

Ce Readme concerne les 2 fichiers jupyter notebook (.ipynb) contenu dans ce dossier.

Ces deux fichiers de code python ont 2 tâches complémentaires mais totalement dissociable.

# Fichier parsing_script.ipynb #

Ce fichier est un fichier générique de parsing de scénarios de films. 
Il est capable de traiter les scénarios au format pdf et txt.

Attention : Ce script parse correctement la plupart des scénarios du moment qu'il respecte les normes d'indentation entre les différentes composantes du scénario
(dialogue, transition caméra, narration, localisation de scènes).

Ce fichier va parser le script et en sortir 3 fichiers différents :
- Le premier correspond à l'ensemble des informations des personnages du script -> charinfo
- Le second correspond à l'ensemble des dialogues identifiés dans le scénario -> dialogue
- Le dernier correspond à l'ensemble des éléments taggés dans le scénario -> tagged

Le troisième type de fichier est celui utilisé lors de l'analyse de film contenu dans le 2ème fichier de code. 

## Lancement du script ##
Le code a été développé afin de parser l'ensemble des scénarios de film stocké au format .txt ou au format .pdf dans le fichier 
test situé au chemin suivant :
data_science/scripts/test

L'ensemble des fichiers parsé seront quant à eux stockés dans le répertoire suivant :
data_science/scripts/parsed

Un troisième fichier comportant les métadonnées (récupérées sur des sites tel que imsdb, dailyscript) au format .json
permet également le parsing du script et la récupération de ces métadonnées se fait également de manière automatique
au lancement du code python.


## Fichier joker(1).ipynb ##

Ce script nécessite l'interaction avec l'utilisateur lors de l'exécution de la 2ème cellule du jupyter notebook.

L'utilisateur doit alors renseigner le nom du script situé dans le répertoire 
data_science/scripts/parsed/tagged
qu'il souhaite analyser.

Il est important de noter que le nom renseigné doit comporter l'extension du fichier (.pdf ou .txt)
Exemple : Joker_parsed.txt
Après cela, les titres peuvent être modifiés en dur dans le code afin de changer les titres des différents graphiques

Il est à noter que ce script a été développé pour un script en particulier, faisant l'objet de la preuve de concept
(Joker), et a été tester sur un petit échantillon de scénario par 


## Lancement du script ##
L'ensemble du script python est automatisé et marche sur un échantillon de validation, certaines corrections mineures
doivent cependant être apporté lors de la phase de nettoyage du fichier texte de parsing.

