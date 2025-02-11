# INF7370-TP1

## Rapport

https://docs.google.com/document/d/1R07RbNdbz36tJplKfWbKk1suP1lfiz0ui6_tNC1VCtk/edit?usp=sharing

## Préparer l'environnement

Créer et activer l'environnement virtuel, puis installer les dépendances.

Linux :

```bash
python -m venv .venv; source .venv/bin/activate; pip install -r requirements.txt
```

Windows :

```bash
python -m venv .venv; source .venv/Scripts/activate; pip install -r requirements.txt
```

Liste des dépendances et justification :

- Levenshtein : Nous calculons la distance entre les mots des tweets des utilisateurs. La distance de Levenshtein est parfaite pour cela et implémenter cette distance en Python est très lent. C'est pourquoi nous utilisons la bibliothèque précompilée en Cython, beaucoup plus rapide.
- Tqdm : PPermet d'afficher une barre de progression pour des tâches relativement longues.
- NumPy : Essentiel pour tout traitement rapide en apprentissage automatique.
- Pandas : Essentiel pour la gestion des données.

## Préparation des données

Pour exécuter cette commande, vous devez préalablement avoir téléchargé les [données brutes](https://ena01.uqam.ca/course/view.php?id=69597&section=1) et décompresser les données dans un dossier "Datasets/" à la racine du projet.

```bash
python features_extraction.py
```

Vous pouvez télécharger les données prétraitées avec cette commande (ou sur ce [lien google drive](https://drive.google.com/file/d/16UjfrzSh00xDWD7f_aeJDRGILbd7Esuk/view?usp=drive_link)) :

Linux

```bash
wget -O preprocessed_data.csv "https://drive.usercontent.google.com/download?id=16UjfrzSh00xDWD7f_aeJDRGILbd7Esuk&export=download&authuser=0&confirm=t&uuid=fe0fb91b-5283-4c1c-a699-22fac24614a9&at=AIrpjvNVtC3gxolaiwIFK5KybET5:1739211978067"
```

Windows

```bash
curl -L "https://drive.usercontent.google.com/download?id=16UjfrzSh00xDWD7f_aeJDRGILbd7Esuk&export=download&authuser=0&confirm=t&uuid=fe0fb91b-5283-4c1c-a699-22fac24614a9&at=AIrpjvNVtC3gxolaiwIFK5KybET5:1739211978067" -o preprocessed_data.csv
```

Cette commande crée un fichier CSV à la racine nommé "preprocessed_data.csv"

## **Description des colonnes**

| Nom de la colonne                 | Type de données | Description                                                                                                      |
| --------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| **user_id**                       | `int`           | Identifiant unique pour chaque utilisateur.                                                                      |
| **created_at**                    | `datetime`      | Horodatage indiquant la date de création du compte Twitter.                                                      |
| **collected_at**                  | `datetime`      | Horodatage indiquant la date de collecte des données utilisateur.                                                |
| **num_followings**                | `int`           | Nombre de comptes suivis par l'utilisateur.                                                                      |
| **num_followers**                 | `int`           | Nombre d'abonnés de l'utilisateur.                                                                               |
| **num_tweets**                    | `int`           | Nombre total de tweets publiés par l'utilisateur.                                                                |
| **length_screen_name**            | `int`           | Longueur du nom d'utilisateur Twitter.                                                                           |
| **length_description**            | `int`           | Longueur de la description du profil utilisateur.                                                                |
| **followings**                    | `string`        | Liste des identifiants des utilisateurs suivis, séparés par des virgules.                                        |
| **variance_of_followings**        | `float`         | Variance des identifiants des utilisateurs suivis, représentant la dispersion des valeurs.                       |
| **tweet**                         | `list`          | Liste des tweets publiés par l'utilisateur.                                                                      |
| **avg_norm_levenshtein_distance** | `float`         | Moyenne de la distance de Levenshtein normalisée entre les tweets de l'utilisateur, mesurant leur similarité.    |
| **z_score_similarity**            | `float`         | Score normalisé (Z-score) de la similarité des tweets de l'utilisateur par rapport aux autres utilisateurs.      |
| **class**                         | `int`           | **Étiquette cible pour la classification :**<br> - `0` : Utilisateur légitime<br> - `1` : Pollueur (compte spam) |

## Algorithmes et évaluation

TODO

## Analyse comparative sur des classes déséquilibrées

TODO
