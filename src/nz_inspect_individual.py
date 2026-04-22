import json

NZ_METADATA_PATH = r"C:\Users\kupak\Desktop\GitHubProjects\thermal_animal_detection\data\raw\nz_thermal\individual-metadata\799183_metadata.json"

with open(NZ_METADATA_PATH, "r") as f:
    data = json.load(f)

print("Top-level keys:", data.keys())

# Print first 2 entries only (avoid massive output)
print("\nSample structure:")
for key in list(data.keys())[:5]:
    print(f"\nKey: {key}")
    print(data[key])