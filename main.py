from loader import load_image, normalize
from Preprocessing.Georeferencing import georeferencing, visualize_georeferencing, visualize_comparison
import cv2
import matplotlib.pyplot as plt
from Preprocessing.intensity_normalisation import intensity_normalization

# Load images
source = load_image(r"C:\Users\JAHANVI\lunar-image-comparison\Preprocessing\SourceImage.png")
reference = load_image(r"C:\Users\JAHANVI\lunar-image-comparison\Preprocessing\ReferenceImage.jpeg")

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

source_normalized = intensity_normalization(
    source,
    low_percentile=2,
    high_percentile=98
)

reference_normalized = intensity_normalization(
    reference,
    low_percentile=2,
    high_percentile=98
)

cv2.imwrite(
    "source_normalized.png",
    source_normalized
)

cv2.imwrite(
    "reference_normalized.png",
    reference_normalized
)

plt.figure(figsize=(12, 6))




# Normalized Source
plt.subplot(2, 2, 2)
plt.imshow(source_normalized, cmap="gray")
plt.title("Normalized Source Image")
plt.axis("off")


# Original Reference
plt.subplot(2, 2, 3)
plt.imshow(reference, cmap="gray")
plt.title("Original Reference Image")
plt.axis("off")


# Normalized Reference
plt.subplot(2, 2, 4)
plt.imshow(reference_normalized, cmap="gray")
plt.title("Normalized Reference Image")
plt.axis("off")


plt.tight_layout()
plt.show()
