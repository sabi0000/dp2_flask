import os
import cv2
import numpy as np
from scipy.stats import entropy
from flask_cors import CORS
import subprocess
from tensorflow import keras

from PIL import Image



from flask_session import Session
from flask import Flask, send_file, render_template, request, session, send_from_directory, abort, url_for, jsonify

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_FILE_DIR'] = './flask_session'
app.config['SECRET_KEY'] = 'your_secret_key'  # Povinné pre session!
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True  # Musí byť True, ak používate 'None'
Session(app)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Absolútna cesta
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
SEGMENTED_FOLDER = os.path.join(os.getcwd(), 'static')
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)
app.config["SEGMENTED_FOLDER"] = SEGMENTED_FOLDER

UPLOAD_FOLDER_MODEL = 'uploads_models'
app.config['UPLOAD_FOLDER_MODEL'] = UPLOAD_FOLDER_MODEL

model = keras.models.load_model("model/skin_cancer_model.h5")

@app.route('/api/message', methods=['GET'])
def get_message():
    return jsonify({"message": "Ahoj z Flask backendu!"})

@app.route("/check_session")
def check_session():
    return jsonify({"session": dict(session)})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)
    
    # Store the image path in the session
    session['image_path'] = image_path
    session.modified = True 
    print(f"File uploaded successfully at {image_path}")  # Debug print
    print(f"Session image_path: {session['image_path']}")  # Debug print
    print(f"Session after setting image_path: {session}")  # Debug print

    return f"File uploaded successfully at {image_path}"

@app.route('/conv_upload', methods=['POST'])
def conv_upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)
    
    # Store the image path in the session
    session['image_path'] = image_path
    session.modified = True 
    print(f"File uploaded successfully at {image_path}")  # Debug print
    print(f"Session image_path: {session['image_path']}")  # Debug print
    print(f"Session after setting image_path: {session}")  # Debug print

    return f"File uploaded successfully at {image_path}"

@app.route('/render', methods=['POST'])
def render_animation():
    print("Rendering requested")  # Debug print
    image_path = session.get('image_path')
    print(f"Session image_path before rendering: {image_path}")  # Debug print

    if not image_path:
        print("Error: No image uploaded.")  # Debug print
        return "Error: No image uploaded.", 400

    # Set the image path as an environment variable
    os.environ['IMAGE_PATH'] = image_path

    # Render the animation using Manim
    result = os.system(f'manim -p -ql animation_canny.py CannyFilterAnimation --disable_caching')
    print(f"Manim exit code: {result}")

    if result != 0:
        print("Rendering failed")  # Debug print
        return "Error: Rendering failed.", 500

    return "Animation rendered!"

@app.route('/sobel_rendering', methods=['POST'])
def render_sobel_animation():
    print("Sobel rendering requested")  # Debug print
    print(f"Session at start of sobel_rendering: {session}")  # Debug print
    print(f"Session keys available: {list(session.keys())}")  # Debug keys


    image_path = session.get('image_path')
    print(f"Session image_path before rendering: {image_path}")  # Debug print
    if not image_path:
        print("Error: No image uploaded.")  # Debug print
        return "Error: No image uploaded.", 400

    os.environ['IMAGE_PATH'] = image_path
    print(f"Image path set to: {image_path}")  # Debug print

    video_path = "media/videos/animation2/480p15/SobelFilterAnimation.mp4"  # Zmeňte na správnu cestu
    if os.path.exists(video_path):
        os.remove(video_path)
        print("Existing video removed.") 

    # Spustite renderovanie a skontrolujte výsledok
    print("Starting rendering...")  # Debug print
    result = os.system(f'manim -p -ql animation2.py SobelFilterAnimation --disable_caching')
    print(f"Manim exit code: {result}")
   
    if result != 0:
        print("Rendering failed")  # Debug print
        return "Error: Rendering failed.", 500

    print("Rendering completed successfully")  # Debug print
    return "Animation rendered!"

@app.route('/simple_render', methods=['POST'])
def simple_animation():
    print("Rendering requested")  # Debug print
    image_path = session.get('image_path')
    print(f"Session image_path before rendering: {image_path}")  # Debug print

    if not image_path:
        print("Error: No image uploaded.")  # Debug print
        return "Error: No image uploaded.", 400

    # Set the image path as an environment variable
    os.environ['IMAGE_PATH'] = image_path

    # Render the animation using Manim
    result = os.system(f'manim -p -ql animation_simpleTreshold.py ThresholdingAnimation --disable_caching')
    print(f"Manim exit code: {result}")

    if result != 0:
        print("Rendering failed")  # Debug print
        return "Error: Rendering failed.", 500

    return "Animation rendered!"

@app.route('/adaptive_render', methods=['POST'])
def adaptive_animation():
    print("Rendering requested")  # Debug print
    image_path = session.get('image_path')
    print(f"Session image_path before rendering: {image_path}")  # Debug print

    if not image_path:
        print("Error: No image uploaded.")  # Debug print
        return "Error: No image uploaded.", 400

    # Set the image path as an environment variable
    os.environ['IMAGE_PATH'] = image_path

    # Render the animation using Manim
    result = os.system(f'manim -p -ql animation_adaptiveTreshold.py AdaptiveAnimation --disable_caching')
    print(f"Manim exit code: {result}")

    if result != 0:
        print("Rendering failed")  # Debug print
        return "Error: Rendering failed.", 500

    return "Animation rendered!"

@app.route('/gauss_render', methods=['POST'])
def gauss_animation():
    print("Rendering requested")  # Debug print
    image_path = session.get('image_path')
    print(f"Session image_path before rendering: {image_path}")  # Debug print

    if not image_path:
        print("Error: No image uploaded.")  # Debug print
        return "Error: No image uploaded.", 400

    # Set the image path as an environment variable
    os.environ['IMAGE_PATH'] = image_path

    # Render the animation using Manim
    result = os.system(f'manim -p -ql animation_gauss.py GaussAnimation --disable_caching')
    print(f"Manim exit code: {result}")

    if result != 0:
        print("Rendering failed")  # Debug print
        return "Error: Rendering failed.", 500

    return "Animation rendered!"

@app.route('/median_render', methods=['POST'])
def median_animation():
    print("Rendering requested")  # Debug print
    image_path = session.get('image_path')
    print(f"Session image_path before rendering: {image_path}")  # Debug print

    if not image_path:
        print("Error: No image uploaded.")  # Debug print
        return "Error: No image uploaded.", 400

    # Set the image path as an environment variable
    os.environ['IMAGE_PATH'] = image_path

    # Render the animation using Manim
    result = os.system(f'manim -p -ql animation_median.py MedianAnimation --disable_caching')
    print(f"Manim exit code: {result}")

    if result != 0:
        print("Rendering failed")  # Debug print
        return "Error: Rendering failed.", 500

    return "Animation rendered!"

@app.route('/network_rendering', methods=['POST'])
def render_network_animation():
    print("Network rendering requested")  # Debug print

    video_path = "media/videos/network_user_anim/1080p60/NeuralNetwork.mp4"  # Zmeňte na správnu cestu
    if os.path.exists(video_path):
        os.remove(video_path)
        print("Existing video removed.") 

    # Spustite renderovanie a skontrolujte výsledok
    print("Starting rendering...")  # Debug print
    result = os.system(f'manim -p -ql network_user_anim.py NeuralNetwork --disable_caching')
    print(f"Manim exit code: {result}")
   
    if result != 0:
        print("Rendering failed")  # Debug print
        return "Error: Rendering failed.", 500

    print("Rendering completed successfully")  # Debug print
    return "Animation rendered!"

@app.route('/acc_rendering', methods=['POST'])
def render_acc_animation():
    print("Network rendering requested")  # Debug print

    # Cesty k videám
    video_paths = [
        "media/videos/animation_acc/480p15/AccuracyLossGraph.mp4",
        "media/videos/animation_valid/1080p60/ValidationAccuracyLossGraph.mp4"
    ]

    for video_path in video_paths:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Existing video removed: {video_path}")

    # Spustite renderovanie a skontrolujte výsledok
    print("Starting rendering...")  # Debug print
    result1 = os.system(f'manim -p -ql animation_acc.py AccuracyLossGraph --disable_caching')
    result2 = os.system(f'manim -p -ql animation_valid.py ValidationAccuracyLossGraph --disable_caching')
    print(f"Manim exit codes: {result1}, {result2}")

    if result1 != 0 or result2 != 0:
        print("Rendering failed")  # Debug print
        return "Error: Rendering failed.", 500

    print("Rendering completed successfully")  # Debug print
    return "Both animations rendered!" 

@app.route('/set_gauss_size', methods=['POST'])
def set_gauss():
    data_gauss = request.get_json()  # Získajte JSON dáta
    gauss_size = data_gauss.get('gauss_size')   # Získajte hodnotu gauss_size
    sigma_size = data_gauss.get('sigma_size')   # Získajte hodnotu sigma_size

    if gauss_size is None or sigma_size is None:
        return "Error: Gauss size and sigma size are required", 400
    
    os.environ['GAUSS_SIZE'] = str(gauss_size)
    os.environ['SIGMA_SIZE'] = str(sigma_size)
    print(f"Gauss size set to {gauss_size}, sigma set to {sigma_size}")  # Debug print

    return f"Gauss size successfully set to {gauss_size}, sigma set to {sigma_size}"

@app.route('/set_ksize', methods=['POST'])
def set_ksize():
    data = request.get_json()  # Získajte JSON dáta
    ksize = data.get('ksize')   # Získajte hodnotu ksize

    if ksize is None:
        return "Error: Kernel size is required", 400
    
    # Set the kernel size in the environment variable
    os.environ['KERNEL_SIZE'] = str(ksize)  # Uistite sa, že je to reťazec
    print(f"Kernel size set to {ksize}")  # Debug print

    return f"Kernel size successfully set to {ksize}"

@app.route('/set_canny', methods=['POST'])
def set_canny():
    data = request.get_json()  # Get all data at once
    lower_threshold = data.get('lower_threshold')
    upper_threshold = data.get('upper_threshold')

    # Check if thresholds are provided
    if lower_threshold is None or upper_threshold is None:
        return "Error: threshold is required", 400

    try:
        # Convert thresholds to integers
        lower_threshold = int(lower_threshold)
        upper_threshold = int(upper_threshold)
    except ValueError:
        return "Error: thresholds must be integers", 400

    # Set environment variables as strings
    os.environ['LOWER_THRESHOLD'] = str(lower_threshold)
    os.environ['UPPER_THRESHOLD'] = str(upper_threshold)

    print(f"Lower threshold set to {lower_threshold}")
    print(f"Upper threshold set to {upper_threshold}")

    return f"Lower threshold successfully set to {lower_threshold} and Upper threshold set to {upper_threshold}"

@app.route('/set_simple_treshold', methods=['POST'])
def set_simple_treshold():
    data = request.get_json()  # Get all data at once
    threshold = data.get('threshold')

    # Check if thresholds are provided
    if threshold is None:
        return "Error: threshold is required", 400

    try:
        # Convert thresholds to integers
        threshold = int(threshold)
    except ValueError:
        return "Error: thresholds must be integers", 400

    # Set environment variables as strings
    os.environ['THRESHOLD'] = str(threshold)


    print(f"threshold set to {threshold}")


    return f"Lower threshold successfully set to {threshold} "

@app.route('/set_adaptive_treshold', methods=['POST'])
def set_adaptive():
    data= request.get_json()  # JSON dáta
    adaptive_treshold = data.get('adaptive_treshold')   
    constant = data.get('constant')   

    if adaptive_treshold is None or constant is None:
        return "Error: adaptive_treshold and constant are required", 400
    
    os.environ['ADAPTIVE_THRESHOLD'] = str(adaptive_treshold)
    os.environ['CONSTANT'] = str(constant)
    print(f"Treshold set to {adaptive_treshold}, constant set to {constant}")  # Debug print

    return f"reshold successfully set to {adaptive_treshold}, constant set to {constant}"

@app.route('/set_neural_network', methods=['POST'])
def set_network():
    data_network = request.get_json()  # Získajte JSON dáta
    layers = data_network.get('layers')   # Získajte hodnotu gauss_size
    neurons = data_network.get('neurons')   # Získajte hodnotu sigma_size

    if layers is None or neurons is None:
        return "Error: Layers and neurons are required", 400
    
    os.environ['LAYERS'] = str(layers)
    os.environ['NEURONS'] = str(neurons)
    print(f"Layers set to {layers}, neurons set to {neurons}")  # Debug print

    return f"Layers successfully set to {layers}, neurons set to {neurons}"

@app.route('/conv_rendering', methods=['POST'])
def conv_animation():
    print("Conv rendering requested")  # Debug print
    print(f"Session at start of conv_rendering: {session}")  # Debug print
    print(f"Session keys available: {list(session.keys())}")  # Debug keys


    image_path = session.get('image_path')
    print(f"Session image_path before rendering: {image_path}")  # Debug print
    if not image_path:
        print("Error: No image uploaded.")  # Debug print
        return "Error: No image uploaded.", 400

    os.environ['IMAGE_PATH'] = image_path
    print(f"Image path set to: {image_path}")  # Debug print

    video_path = "media/videos/mnist_user/1920p30/ImageDecompositionAnimationUser.mp4"  # Zmeňte na správnu cestu
    if os.path.exists(video_path):
        os.remove(video_path)
        print("Existing video removed.") 

    # Spustite renderovanie a skontrolujte výsledok
    print("Starting rendering...")  # Debug print
    result = os.system(f'manim -p -ql mnist_user.py ImageDecompositionAnimationUser --disable_caching')
    print(f"Manim exit code: {result}")
   
    if result != 0:
        print("Rendering failed")  # Debug print
        return "Error: Rendering failed.", 500

    print("Rendering completed successfully")  # Debug print
    return "Animation rendered!"

@app.route('/process_image', methods=['POST'])
def process_image():
    num_edges = np.int32(1234)  # Example int32 type
    laplacian_var = np.float32(150.5)  # Example float32 type
    edge_density = np.float32(0.2)
    edge_entropy = np.float32(2.7)

    # Convert to standard Python int or float types
    num_edges = int(num_edges)
    laplacian_var = float(laplacian_var)
    edge_density = float(edge_density)
    edge_entropy = float(edge_entropy)

    #image_file = request.files['image']
    gauss_size = int(request.form['gauss_size'])
    lower_threshold = int(request.form['lower_threshold'])
    upper_threshold = int(request.form['upper_threshold'])
    sigma=int(request.form['sigma'])

    # Save the uploaded image to a temporary file
    image_path = os.path.join('static', 'ISIC_0024311.jpg')
    #image_file.save(image_path)

    # Load and process the image with OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, (gauss_size, gauss_size), 0)
    edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)
    num_edges = np.sum(edges > 0)
    laplacian_var = cv2.Laplacian(edges, cv2.CV_64F).var()
    edge_density = num_edges / (image.shape[0] * image.shape[1])
    histogram, _ = np.histogram(edges.flatten(), bins=256, range=[0, 256])
    edge_entropy = entropy(histogram)

    num_edges = int(num_edges)
    laplacian_var = round(laplacian_var, 4)
    edge_density = round(edge_density, 4)
    edge_entropy = round(edge_entropy, 4)

    # Save the processed image
    processed_image_path = os.path.join('static', 'processed_image.png')
    cv2.imwrite(processed_image_path, edges)

    # Return the template with the processed image URL
    #return render_template('hrany3.html', process_image=True, processed_image_url=url_for('static', filename='processed_image.png'))
    return render_template(
        'hrany3.html',
        process_image=True,
        processed_image_url=url_for('static', filename='processed_image.png'),
        num_edges=num_edges,
        laplacian_var=laplacian_var,
        edge_density=edge_density,
        edge_entropy=edge_entropy
    )


@app.route('/compare_image_data', methods=['POST'])
def compare_image_data():
    data = request.json
    current_metrics = data.get('current_metrics', {})

    REFERENCE_METRICS = {
        "num_edges": 15467,
        "laplacian_var": 37868.1516,
        "edge_density": 0.0572,
        "edge_entropy": 0.2194
    }

    # Example of previous metrics that may be NumPy types
    previous_metrics = {
        "num_edges": np.int32(1000),        # Example numpy int32 type
        "laplacian_var": np.float32(150.0), # Example numpy float32 type
        "edge_density": np.float32(0.1),
        "edge_entropy": np.float32(2.7)
    }

    # Convert all NumPy values to standard Python int or float
    previous_metrics = {
        k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
        for k, v in previous_metrics.items()
    }

    # Ensure current metrics are also converted
    current_metrics = {
        "num_edges": int(current_metrics.get("num_edges", 0)),
        "laplacian_var": float(current_metrics.get("laplacian_var", 0.0)),
        "edge_density": float(current_metrics.get("edge_density", 0.0)),
        "edge_entropy": float(current_metrics.get("edge_entropy", 0.0))
    }

    # Comparison logic
    comparison_results = {
        "num_edges_match": abs(current_metrics["num_edges"] - REFERENCE_METRICS["num_edges"]) <= 10000,
        "laplacian_var_match": abs(current_metrics["laplacian_var"] - REFERENCE_METRICS["laplacian_var"]) <= 500,
        "edge_density_match": abs(current_metrics["edge_density"] - REFERENCE_METRICS["edge_density"]) <= 0.01,
        "edge_entropy_match": abs(current_metrics["edge_entropy"] - REFERENCE_METRICS["edge_entropy"]) <= 0.01
    }

    return jsonify(comparison_results)

@app.route("/segment", methods=["POST"])
def segment():
    data = request.get_json()
    r, g, b = data["color"]["r"], data["color"]["g"], data["color"]["b"]

    # Fixný obrázok
    image_path = "static/ISIC_0024311.jpg"
    image = cv2.imread(image_path)
    
    # Konverzia do HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]

    # Definovanie tolerancie
    lower_bound = np.array([max(0, color_hsv[0] - 10), 50, 50])
    upper_bound = np.array([min(180, color_hsv[0] + 10), 255, 255])

    # Thresholding
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Uloženie segmentovaného obrázka do static/
    output_filename = "segmented.png"
    output_path = os.path.join(app.config["SEGMENTED_FOLDER"], output_filename)
    cv2.imwrite(output_path, result)

    # Flask automaticky servíruje súbory zo static/, takže môžeš vrátiť URL
    return jsonify({"image_url": f"/static/{output_filename}"})

@app.route('/set_metrics', methods=['POST'])
def set_metrics():
    data_metrics = request.get_json()  
    layers_m = data_metrics.get('layers_m')  
    neurons_m = data_metrics.get('neurons_m')
    epochs_m=data_metrics.get('epochs_m')  

    if layers_m  is None or neurons_m is None or epochs_m is None:
        return "Error: Layers and neurons are required", 400
    
    os.environ['LAYERS_M'] = str(layers_m)
    os.environ['NEURONS_M'] = str(neurons_m)
    os.environ['EPOCHS_M'] = str(epochs_m)
    print(f"Layers set to {layers_m}, neurons set to {neurons_m}, epochs set to {epochs_m}")  # Debug print

    return f"Layers successfully set to {layers_m}, neurons set to {neurons_m }, epochs set to {epochs_m}"

labels = {0: 'AKIEC', 1: 'BCC', 2: 'BKL', 3: 'DF', 4: 'MEL', 5: 'NV', 6: 'VASC'}

def preprocess_image(filepath):
    # Load the image using PIL
    img = Image.open(filepath)
    # Resize the image to the required size (e.g., 224x224)
    img = img.resize((64, 64))
    # Convert to numpy array
    img_array = np.array(img)
    # Normalize pixel values between 0 and 1
    img_array = img_array / 255.0
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    filepath = os.path.join('uploads', file.filename)  # Ensure this folder exists
    file.save(filepath)

    # Preprocess the image
    img_array = preprocess_image(filepath)

    # Po načítaní obrázka a predspracovaní
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # --> napr. 4

    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = prediction[0][predicted_index]  # Tá najvyššia pravdepodobnosť

    # Premapovanie čísla triedy na názov
    predicted_label = labels[predicted_class]

    print(f"Predikovaná trieda: {predicted_label} a istota {confidence}")

    return jsonify({
    "label": predicted_label,
    "confidence": float(confidence)  # napr. 0.85
}) 

@app.route('/video')
def video():
   return send_file('media/videos/animation/480p15/BinaryFilterAnimation.mp4')

@app.route('/video/binary')
def binary_video():
    return send_file('media/videos/animation/480p15/BinaryFilterAnimation.mp4')

@app.route('/video/canny')
def canny_video():
    return send_file('media/videos/animation_canny/480p15/CannyFilterAnimation.mp4')

@app.route('/video/sobel')
def sobel_video_example():
    return send_file('media/videos/animation2/480p15/SobelFilterAnimation.mp4')

@app.route('/video/canny_example')
def canny_video_example():
    return send_file('media/videos/animation_canny/480p15/CannyFilterAnimationExample.mp4')

@app.route('/video/sobel_example')
def sobel_video():
    return send_file('media/videos/animation2/480p15/SobelFilterAnimationExample.mp4')

@app.route('/video/ai_neuron')
def neuron_video():
    return send_file('media/videos/animation_neuron/1080p60/ArtificialNeuron.mp4')

@app.route('/video/bio_neuron')
def bio_neuron_video():
    return send_file('media/videos/animation_nuron2/videoplayback.mp4')

@app.route('/video/neural_network')
def network_video():
    return send_file('media/videos/network_user_anim/480p15/NeuralNetwork.mp4')

@app.route('/video/sigmoid')
def sigmoid_video():
    return send_file('media/videos/animation_activation/1080p60/NeuralNetworkActivation.mp4')

@app.route('/video/relu')
def relu_video():
    return send_file('media/videos/animation_Relu/1080p60/ReLUActivation.mp4')

@app.route('/video/tanh')
def tanh_video():
    return send_file('media/videos/animation_tanh/1080p60/TanhActivation.mp4')

@app.route('/video/softmax')
def softmax_video():
    return send_file('media/videos/animation_softmax/1080p60/SoftmaxActivation.mp4')

@app.route('/video/brain')
def brain_video():
    return send_file('media/videos/animation_networks_example/1080p60/brain.mp4')

@app.route('/video/networks_example')
def networks_example_video():
    return send_file('media/videos/network_animation/1080p60/NeuralNetwork.mp4')

@app.route('/video/epochs')
def epochs_video():
    return send_file('media/videos/animation_network/1080p60/NeuralNetworkLearning.mp4')

@app.route('/video/acc')
def acc_video():
    return send_file('media/videos/animation_acc/480p15/AccuracyLossGraph.mp4')

@app.route('/video/val')
def val_video():
    return send_file('media/videos/animation_valid/480p15/ValidationAccuracyLossGraph.mp4')

@app.route('/video/cnn')
def cnn_video():
    return send_file('media/videos/animation_networks_example/1080p60/CNNArchitecture.mp4')

@app.route('/video/densenet')
def densenet_video():
    return send_file('media/videos/animation_densenet/1080p60/DenseNet121Dots.mp4')

@app.route('/video/resnet')
def resnet_video():
    return send_file('media/videos/animation_resnet/1080p60/ResNet50Dots.mp4')

@app.route('/video/conv_example')
def conv_example_video():
    return send_file('media/videos/mnist/1920p30/ImageDecompositionAnimation.mp4')

@app.route('/video/conv_user')
def conv_user_video():
    return send_file('media/videos/mnist_user/1920p30/ImageDecompositionAnimationUser.mp4')

@app.route('/video/matrix')
def matrix_video():
    return send_file('media/videos/animation_matrix/1080p60/ConvolutionAnimation.mp4')

@app.route('/video/simple_treshold')
def simple_treshold_video():
    return send_file('media/videos/animation_simpleTreshold/480p15/ThresholdingAnimation.mp4')

@app.route('/video/simple_treshold_exaple')
def simple_treshold_example_video():
    return send_file('media/videos/animation_simpleTreshold/480p15/ThresholdingAnimationExample.mp4')

@app.route('/video/adaptive_treshold')
def adaptive_treshold_video():
    return send_file('media/videos/animation_adaptiveTreshold/480p15/AdaptiveAnimation.mp4')

@app.route('/video/adaptive_treshold_example')
def adaptive_treshold_example_video():
    return send_file('media/videos/animation_adaptiveTreshold/480p15/AdaptiveAnimationExample.mp4')

@app.route('/video/gauss')
def gauss_video():
    return send_file('media/videos/animation_gauss/480p15/GaussAnimation.mp4')

@app.route('/video/gauss_example')
def gauss_example_video():
    return send_file('media/videos/animation_gauss/480p15/GaussAnimationExample.mp4')

@app.route('/video/median')
def median_video():
    return send_file('media/videos/animation_median/480p15/MedianAnimation.mp4')

@app.route('/video/median_example')
def median_example_video():
    return send_file('media/videos/animation_median/480p15/MedianAnimationExample.mp4')

@app.route('/image1')
def image1():
    return send_file('static/ISIC_0024311.jpg')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hrany/<int:page_number>')
def hrany(page_number):
    if page_number == 1:
        return render_template('hrany.html')
    elif page_number == 2:
        return render_template('hrany2.html')
    elif page_number == 3:
        return render_template('hrany3.html')
    else:
        return "Page not found", 404
    #return render_template('hrany.html')

#if __name__ == '__main__':    
#    app.run(debug=True)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)