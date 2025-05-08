from flask import Flask, render_template, request, send_file, jsonify
import os
import torch.optim as optim
from nst import run_style_transfer, load_model, transform_image
from torchvision import transforms
from PIL import Image
from io import BytesIO

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = load_model()
content_layers = ['21']  # relu4_2
style_layers = ['0', '5', '10', '19', '28']  # relu1_1 to relu5_1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apply_style', methods=['POST'])
def apply_style():
    content_file = request.files['content_image']
    style_file = request.files['style_image']

    content_path = os.path.join(UPLOAD_FOLDER, 'content.jpg')
    style_path = os.path.join(UPLOAD_FOLDER, 'style.jpg')
    output_path = os.path.join(OUTPUT_FOLDER, 'styled.jpg')

    content_file.save(content_path)
    style_file.save(style_path)

    # Initialize the image transformation
    content_img = transform_image(content_path)
    target_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([target_img], lr=0.003)

    # Start style transfer and handle asynchronously
    styled_img, _ = run_style_transfer(content_path, style_path, model, content_layers, style_layers, optimizer)

    to_pil = transforms.ToPILImage()
    output_image = to_pil(styled_img.squeeze().cpu().detach())
    output_image.save(output_path)

    # Return path for frontend to display the image
    return jsonify({'status': 'Success', 'image_url': f'/{output_path}'})

if __name__ == '__main__':
    app.run(debug=True)
