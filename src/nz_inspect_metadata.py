import json
from typing import Counter

NZ_METADATA_PATH = r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection\data\raw\nz_thermal\new-zealand-wildlife-thermal-imaging.json"


with open(NZ_METADATA_PATH, "r") as f:
    data = json.load(f)

clips = data["clips"]

species_counter = Counter()

for clip in clips:
    labels = clip.get("labels", [])  # Safe access

    for label in labels:
        species_counter[label] += 1

print("\nSpecies distribution:")
for species, count in species_counter.most_common():
    print(f"{species}: {count}")