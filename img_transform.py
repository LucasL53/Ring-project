from PIL import Image
from torchvision import transforms

# Loads image at image_path, tranforms, and returns as raw array
def load(image_path):
    img = Image.open(image_path).convert('RGB')
    dino_transform = transforms.Compose([
        transforms.Pad((0, 21)),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_t = dino_transform(img).unsqueeze(0)
    print(img_t.shape)
    return img_t.tolist()
