from loader import load_image, normalize
from Preprocessing.Georeferencing import georeferencing, visualize_georeferencing, visualize_comparison


# Load images
source = load_image("data/source.png")
reference = load_image("data/reference.png")

print(f"✓ Source image loaded: {source.shape}")
print(f"✓ Reference image loaded: {reference.shape}")

# Normalize images
source = normalize(source)
reference = normalize(reference)
print("✓ Images normalized to [0, 1]")

# Apply georeferencing
print("\nApplying georeferencing...")
source_georeferenced = georeferencing(source, reference)

print("✓ Georeferencing complete")
print(f"  Source shape: {source.shape} → {source_georeferenced.shape}")
print(f"  Reference shape: {reference.shape}")

# Visualize results
print("\nGenerating visualization...")
visualize_georeferencing(source, reference, source_georeferenced)

# Show overlay comparison
print("Showing overlay comparison...")
visualize_comparison(reference, source_georeferenced, alpha=0.5)

