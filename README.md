# INF7370-TP1

## Rapport

https://www.overleaf.com/project/67ab97206002333b4928d4ee

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

Vous pouvez télécharger les données prétraitées avec cette commande (ou sur ce [lien google drive](https://drive.google.com/drive/folders/1InCDKGSWU_g6GxRgGTLD1rFVXgwWVt0F?usp=drive_link)) :

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
| **num_followings**                | `int`           | Nombre de comptes suivis par l'utilisateur.                                                                      |
| **num_followers**                 | `int`           | Nombre d'abonnés de l'utilisateur.                                                                               |
| **length_screen_name**            | `int`           | Longueur du nom d'utilisateur Twitter.                                                                           |
| **length_description**            | `int`           | Longueur de la description du profil utilisateur.                                                                |
| **avg_time_between_tweets**       | `float`         | Temps moyen en secondes entre 2 tweets.                                                                          |
| **max_time_between_tweets**       | `float`         | Temps maximal en secondes entre 2 tweets.                                                                        |
| **class**                         | `int`           | **Étiquette cible pour la classification :**<br> - `0` : Utilisateur légitime<br> - `1` : Pollueur (compte spam) |
| **variance_of_followings**        | `float`         | Variance des identifiants des utilisateurs suivis, représentant la dispersion des valeurs.                       |
| **avg_norm_levenshtein_distance** | `float`         | Moyenne de la distance de Levenshtein normalisée entre les tweets de l'utilisateur, mesurant leur similarité.    |
| **z_score_similarity**            | `float`         | Score normalisé (Z-score) de la similarité des tweets de l'utilisateur par rapport aux autres utilisateurs.      |
| **account_lifetime_days**         | `int`           | Durée de vie du compte en jours.                                                                                 |
| **following_follower_ratio**      | `float`         | Rapport following/followers.                                                                                     |
| **tweets_per_day**                | `float`         | Nombre de tweets par jour.                                                                                       |
| **mentions_ratio**                | `float`         | Rapport de mentions par tweet.                                                                                   |
| **url_ratio**                     | `float`         | Rapport d'URL par tweet.                                                                                         |


## Algorithmes et évaluation

Pour débuter l'entraînement et l'évaluation des 5 modèles avec des classes équilibrées et déséquilibrées, on utilise cette commande :

```bash
python comparison_all_algorithms.py
```

L'exécution crée des fichiers dans les dossiers `output/models` et `output/graphs`.

Les 6 algorithmes sont :

1. Arbre de décision
2. Bagging
3. AdaBoost
4. Boosting de gradient (GBoost)
5. Forêts d’arbres aléatoires
6. Classification bayésienne naïve

Les données en entrée contiennent **19 251** utilisateurs légitimes et **20 645** utilisateurs pollueurs. Le test **"classes équilibrées"** diminue le nombre d'utilisateurs pollueurs à 5 % du nombre original (**20 645 × 0.05 = 1 032** utilisateurs pollueurs).
