import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# --- LOAD VGG19 ---
def load_model():
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg

# --- TRANSFORM IMAGE ---
def transform_image(image_path, image_size=256):  # reduced to speed up
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# --- FEATURES EXTRACTION ---
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# --- GRAM MATRIX FOR STYLE ---
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# --- MAIN NST FUNCTION ---
def run_style_transfer(content_path, style_path, model, content_layers, style_layers, optimizer, num_steps=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    content = transform_image(content_path).to(device)
    style = transform_image(style_path).to(device)
    target = content.clone().requires_grad_(True).to(device)

    content_features = get_features(content, model, content_layers)
    style_features = get_features(style, model, style_layers)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    style_weight = 1e6
    content_weight = 1e0

    print("🔁 Starting style transfer...")
    for step in range(num_steps):
        optimizer.zero_grad()
        target_features = get_features(target, model, set(content_layers + style_layers))

        content_loss = 0
        for layer in content_layers:
            content_loss += torch.mean((target_features[layer] - content_features[layer])**2)

        style_loss = 0
        for layer in style_layers:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            style_loss += torch.mean((target_gram - style_gram)**2)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}: Total Loss = {total_loss.item():.4f}")

    return target.detach(), total_loss.item()
