from loader import load_image, normalize, visulize


source = load_image("data/source.png")
reference = load_image("data/reference.png")

source = normalize(source)
reference = normalize(reference)

