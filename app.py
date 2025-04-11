from flask import Flask, request, jsonify, render_template, send_file
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import datetime
import cv2
import numpy as np
from color_score import detect_colors_from_pil
from hyper_tune import UNet, DoubleConv  
from abcd_analysis import compute_asymmetry, compute_border_irregularity, compute_diameter  # adjust path


app = Flask(__name__, static_folder='static', template_folder='templates')

# Add the custom classes to safe globals
torch.serialization.add_safe_globals([UNet, DoubleConv])

# Load your PyTorch segmentation model with weights_only=False
model_path = "segmentation_model.pth"
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
model.eval()

# Define input image transformation; adjust input_size and normalization as required.
input_size = (256, 256)
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def segment_image(input_image):
    """
    Preprocess the input PIL image, run the segmentation model,
    and convert the output to a PIL Image.
    """
    # Transform the image to a tensor with a batch dimension.
    img_tensor = transform(input_image).unsqueeze(0)
    
    # Run the segmentation model.
    with torch.no_grad():
        output = model(img_tensor)
    
    # Assume the model returns a tensor of shape [1, 1, H, W] as a probability map.
    if output.dim() == 4 and output.size(1) == 1:
        mask = output.squeeze(0).squeeze(0)
    elif output.dim() == 4:
        # If multiple channels, take the argmax across the channel dimension.
        mask = torch.argmax(output, dim=1).squeeze(0)
    else:
        mask = output
    
    # Convert the output tensor to a NumPy array.
    mask_np = mask.cpu().numpy()
    
    # If output is a probability map (not uint8), threshold at 0.5 to obtain a binary mask.
    if mask_np.dtype != 'uint8':
        mask_np = (mask_np > 0.5).astype('uint8') * 255

    # Return the segmentation result as a PIL image.
    segmented_image = Image.fromarray(mask_np)
    return segmented_image

@app.route('/')
def index():
    # Render your unchanged UI (index.html).
    return render_template('index.html')


@app.route('/generate-report', methods=['POST'])
def generate_report():
    data = request.json
    A = float(data.get("asymmetry", 0))
    B = float(data.get("border", 0))
    C = float(data.get("color_score", 0))
    D = float(data.get("diameter_score", 0))
    diagnosis = data.get("diagnosis", "Unspecified")

    tds = A * 1.3 + B * 0.1 + C * 0.5 + D * 0.5

    # Reuse saved segmented + original image path if tracked, or hardcode for demo
    segmented_path = "segmented_outputs/last_segmented.png"
    original_path = "static/latest_uploaded.png"

    # Output path
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # (Insert same PDF generation code here as from the previous message)
    # Replace hardcoded values with A, B, C, D, and use segmented_path/original_path

    # Save and send PDF
    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="DermaScan_Report.pdf", mimetype='application/pdf')


@app.route('/segment', methods=['POST'])
def segment_endpoint():
    # Check if a file was uploaded.
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "Empty file name"}), 400

    try:
        # Open the uploaded image.
        input_image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image file"}), 400

    # Run segmentation.
    segmented_image = segment_image(input_image)
    
    # Save the segmented image locally in 'segmented_outputs'
    save_folder = os.path.join("segmented_outputs")
    os.makedirs(save_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"segmented_{timestamp}.png"
    output_path = os.path.join(save_folder, output_filename)
    segmented_image.save(output_path)

    # Prepare the segmented image to send in the response.
    buffer = io.BytesIO()
    segmented_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Convert PIL segmented image to grayscale OpenCV format
    mask_np = np.array(segmented_image.convert("L"))  # grayscale

    # Binarize mask
    _, bin_mask = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)

    # Run feature extraction
    A = compute_asymmetry(bin_mask)
    B = compute_border_irregularity(bin_mask)
    D = compute_diameter(bin_mask)

    # Save the segmented image locally
    segmented_image.save(output_path)

    # Prepare image buffer

    # Encode image to base64
    import base64
    present_colors, C = detect_colors_from_pil(input_image)

    # Encode image
    buffer = io.BytesIO()
    segmented_image.save(buffer, format="PNG")
    buffer.seek(0)

    segmented_buffer = io.BytesIO()
    segmented_image.save(segmented_buffer, format="PNG")
    segmented_buffer.seek(0)
    segmented_base64 = base64.b64encode(segmented_buffer.getvalue()).decode('utf-8')

    # Encode original image
    original_buffer = io.BytesIO()
    input_image.save(original_buffer, format="PNG")
    original_buffer.seek(0)
    original_base64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')
    D_raw, D_score = compute_diameter(bin_mask)
    # Return both in JSON
    return jsonify({
        "original_image": original_base64,
        "segmented_image": segmented_base64,
        "asymmetry": round(A, 4), 
        "border": round(B, 4),
        "diameter": round(D_raw, 2),
        "diameter_score": round(D_score, 3),
        "color_score": round(C, 2),
        "present_colors": present_colors,
        "filename": output_filename
    })


if __name__ == '__main__':
    app.run(debug=True)
