# BloodPrint: Blood Group Detection from Fingerprint

A deep learning web application that predicts blood groups from fingerprint images using Convolutional Neural Networks (CNN).

## Features

- Predict blood group from fingerprint images
- Interactive web interface with beautiful UI/UX
- Drag and drop file upload
- Real-time result visualization with charts
- Responsive design for all devices
- Blood group information and facts

## Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for model training and inference
- **Flask**: Web application framework
- **HTML/CSS/JavaScript**: Front-end development
- **Bootstrap**: UI framework for responsive design
- **Chart.js**: Interactive charts for result visualization
- **Particles.js**: Background animation effects
- **Animate.css**: UI animations

## Project Structure

```
BloodPrint/
├── app.py                    # Flask application
├── blood_group_model_final.h5 # Trained CNN model
├── templates/
│   ├── index.html           # Main page
│   └── about.html           # About page
├── static/
│   ├── css/
│   │   └── style.css        # CSS styles
│   ├── js/
│   │   └── script.js        # JavaScript
│   └── images/              # Images
├── uploads/                 # Uploaded images (created automatically)
└── dataset_blood_group/     # Training dataset
    ├── A+/
    ├── A-/
    ├── B+/
    ├── B-/
    ├── AB+/
    ├── AB-/
    ├── O+/
    └── O-/
```

## Installation and Setup

1. **Clone this repository**:
   ```
   git clone https://github.com/yourusername/bloodprint.git
   cd bloodprint
   ```

2. **Create a virtual environment**:
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

5. **Train the model** (optional - if you don't already have a trained model):
   - Run the provided Jupyter notebook to train the model
   - Save the model as 'blood_group_model_final.h5' in the project root

6. **Run the application**:
   ```
   python app.py
   ```

7. **Access the web application**:
   Open a web browser and go to `http://127.0.0.1:5000/`

## Requirements

```
flask==2.0.1
tensorflow==2.8.0
numpy==1.22.3
opencv-python==4.5.5.64
werkzeug==2.0.1
```

## Training Your Own Model

To train your own model, follow these steps:

1. Organize fingerprint images in dataset_blood_group directory with blood group subfolders
2. Run the Jupyter notebook provided in the repository
3. Save the trained model as 'blood_group_model_final.h5'

## Model Architecture

The CNN model uses the following architecture:
- 4 convolutional layers with ReLU activation
- Max pooling after each convolutional layer
- Dropout regularization to prevent overfitting
- Dense layers for classification

## Usage

1. Upload a fingerprint image using the web interface
2. The system will process the image and predict the blood group
3. Results will be displayed with confidence scores and relevant information

## Important Note

This application is for educational and research purposes only. Blood group prediction from fingerprints is an area of ongoing research, and the results should not replace traditional blood typing methods in medical settings.

## License

MIT License

## Credits

- Developed by Your Name/Team
- Fingerprint dataset from [Source]

## Screenshots

[Add screenshots of your application here] 