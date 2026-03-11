# Setup and Execution Instructions

Welcome to the Malnutrition Detection System! Follow this simplified step-by-step guide to download, install, and run the project from scratch.

## Step 1: Download the Project
1. Download the project zip file or clone the repository to your local machine.
2. Extract the contents (if downloaded as a zip) to a folder, for example: `C:\Users\YourUsername\Desktop\detect`.

## Step 2: Open Terminal / Command Prompt
Open your preferred terminal (Command Prompt, PowerShell, or VS Code Terminal) and navigate to the project directory:
```bash
cd C:\Users\YourUsername\Desktop\detect
```

## Step 3: Set Up a Virtual Environment 
Creating a virtual environment ensures that the project dependencies do not interfere with other Python projects on your computer.
```bash
# Create a virtual environment named "myenv"
python -m venv myenv

# Activate the virtual environment
# On Windows:
myenv\Scripts\activate
# On Mac/Linux:
source myenv/bin/activate
```
*(You should see `(myenv)` appear at the beginning of your terminal prompt indicating it is active.)*

## Step 4: Install Dependencies
Run the following command to install all necessary Python packages required by the project:
```bash
python -m pip install --upgrade pip

# Install required packages (from the configs/requirements.txt if available, or manually)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gradio pandas numpy pillow huggingface-hub tqdm scikit-learn
pip install llama-cpp-python --prefer-binary
```

## Step 5: Download Required AI Models
The application relies on the **Mistral 7B** Large Language Model for generating health advisories. We have provided a script to automatically download it to the correct folder.
```bash
python organise/download_mistral.py
```
*(Note: The model is ~4.37 GB and might take 5-20 minutes depending on your internet connection.)*

## Step 6: Train the Model (Optional)
**If you already have the `malnutrition_model.pth` in your `models/` folder, you can SKIP this step.**
To train the ResNet-50 detection model on the local dataset:
```bash
# Label the images automatically
python organise/auto_label_images.py

# Train the model
python training/train_with_labels.py

# Evaluate the model
python evaluate_model.py
```

## Step 7: Launch the Web Application
Now that everything is set up, you can start the application!
```bash
python ui/gradio_app.py
```

## Step 8: Use the App
1. After running the launch command, wait for the terminal to display `Access the app at: http://127.0.0.1:7860`.
2. Open your web browser (Chrome, Edge, Safari, etc.) and go to **http://127.0.0.1:7860**.
3. Upload an image, view the detection results, read the AI's health advisory, and interact with the Q&A section!

To stop the web application, go back to your terminal and press `Ctrl + C`.
