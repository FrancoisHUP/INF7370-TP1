import Levenshtein

tweet1 = "'I Need A Hug'"
tweet2 = "im backkkk!!!!"

dist = Levenshtein.distance(tweet1, tweet2)
normalixed_dist = dist / max(len(tweet1), len(tweet2)) if max(len(tweet1), len(tweet2)) > 0 else 0  # Avoid division by zero
print(f"Levenshtein distance : {dist} | Normalized distance : {normalixed_dist:.2f}")


tweet3 = 'THE BURLESQUE BOOTCAMP SYDNEY - Open Date tickets now available from http://bbootcampsyd.eventbrite.com/ for Jan /... http://fb.me/3rzipF0'
tweet4 = 'THE BURLESQUE BOOTCAMP SYDNEY - Open Date tickets now available from http://bbootcampsyd.eventbrite.com/ for Jan /... http://bit.ly/1v5hvb'

dist = Levenshtein.distance(tweet3, tweet4)
normalixed_dist = dist / max(len(tweet3), len(tweet4)) if max(len(tweet3), len(tweet4)) > 0 else 0  # Avoid division by zero

print(f"Levenshtein distance : {dist} | Normalized distance : {normalixed_dist:.2f}")

