#!/usr/bin/env python3
"""
Malnutrition Detection Web App with Mistral 7B Integration
Complete Gradio interface for malnutrition detection and health advisory generation
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Avoid UnicodeEncodeError on Windows consoles using legacy encodings.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
from src.llm_handler import get_llm

print("\n" + "="*70)
print("Malnutrition Detection Web App - Startup")
print("="*70)

# ============================================================================
# 1. DEVICE SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")

# ============================================================================
# 2. LOAD DETECTION MODEL
# ============================================================================

print("\n✓ Loading malnutrition detection model...")

model_path = os.path.join(project_root, 'models', 'malnutrition_model.pth')

if not os.path.exists(model_path):
    print(f"✗ Error: Model not found at {model_path}")
    print("\nPlease train the model first:")
    print("  python train.py")
    sys.exit(1)

# ResNet-50 with Dropout + Linear classifier head
# Must match the architecture in training/train_with_labels.py
detection_model = models.resnet50()
detection_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),
    torch.nn.Linear(detection_model.fc.in_features, 2)
)
detection_model.load_state_dict(torch.load(model_path, map_location=device))
detection_model = detection_model.to(device)
detection_model.eval()

print(f"✓ Detection model loaded from: {model_path}")

# ============================================================================
# 3. LOAD LLM (MISTRAL 7B)
# ============================================================================

print("\n✓ Loading Mistral 7B language model...")
print("(This may take 10-15 seconds on first run)\n")

try:
    llm = get_llm()
    if llm and llm.llm is not None:
        print("✓ Mistral 7B LLM loaded successfully")
    else:
        reason = getattr(llm, "unavailable_reason", None)
        if reason:
            print(f"✗ LLM unavailable: {reason}")
        else:
            print("✗ LLM initialization failed - check the error messages above")
except Exception as e:
    import traceback
    print(f"✗ Error loading LLM: {str(e)}")
    print(f"Detailed error: {traceback.format_exc()}")
    llm = None

# ============================================================================
# 4. IMAGE PREPROCESSING
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("✓ Image preprocessing pipeline ready")

# ============================================================================
# 5. PREDICTION FUNCTION
# ============================================================================

def predict_and_advise(image):
    """
    Main prediction function
    Takes an image and returns:
    1. Detection result (Healthy/Malnourished with confidence)
    2. Health advisory from Mistral 7B
    """

    if image is None:
        return "No image uploaded", "Please upload an image to get predictions", {}

    try:
        # Preprocess image
        img = transform(image).unsqueeze(0).to(device)

        # Run detection
        with torch.no_grad():
            output = detection_model(img)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item() * 100

        # Format prediction result
        if pred == 1:
            label = "🚨 MALNOURISHED"
            prediction_text = f"{label}\nConfidence: {conf:.2f}%\n\nThis child shows signs of malnutrition and requires attention."
            emoji = "⚠️"
        else:
            label = "✅ HEALTHY"
            prediction_text = f"{label}\nConfidence: {conf:.2f}%\n\nThis child appears to be in good health."
            emoji = "👍"

        # Generate health advisory from LLM
        if llm and llm.llm is not None:
            try:
                advisory = llm.generate_health_advisory(pred, conf)
            except Exception as e:
                print(f"Error generating advisory: {str(e)}")
                advisory = "Error generating health advisory. Please try again."
        else:
            reason = getattr(llm, "unavailable_reason", None) if llm else None
            advisory = (
                f"LLM not available: {reason}\n\n"
                "Please ensure:\n"
                "1. Mistral model file exists in models/ directory\n"
                "2. llama-cpp-python is installed in this environment\n"
                "3. Python version is 3.10-3.12 for llama-cpp-python compatibility\n"
                "4. Sufficient system memory is available"
            )

        # Format advisory
        formatted_advisory = f"{emoji}\n{advisory}"

        # Prepare detailed results
        results = {
            "Prediction": label,
            "Confidence": f"{conf:.2f}%",
            "Status": "Malnourished" if pred == 1 else "Healthy"
        }

        return prediction_text, formatted_advisory, results

    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        return error_msg, "Could not generate advisory", {}

# ============================================================================
# 6. Q&A FUNCTION
# ============================================================================

def answer_question(question):
    """
    Answer questions about malnutrition using Mistral 7B
    """

    if not question or question.strip() == "":
        return "Please enter a question."

    if llm and llm.llm is not None:
        try:
            answer = llm.answer_question(question)
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    else:
        reason = getattr(llm, "unavailable_reason", None) if llm else None
        if reason:
            return f"LLM not available: {reason}"
        return "LLM not available. Please check Mistral 7B installation."

# ============================================================================
# 7. BUILD GRADIO INTERFACE
# ============================================================================

print("\n" + "="*70)
print("Building Gradio Interface")
print("="*70)

with gr.Blocks(title="Malnutrition Detection System", theme=gr.themes.Soft()) as demo:

    # ====================================================================
    # HEADER
    # ====================================================================
    gr.Markdown("# 🍎 Malnutrition Detection & Health Advisory System")
    gr.Markdown(
        """
        This AI-powered system detects malnutrition from facial images and provides 
        health recommendations using advanced computer vision and natural language AI.

        **How to use:**
        1. Upload a child's facial image
        2. Get instant malnutrition detection with confidence score
        3. Receive AI-generated health advisory and recommendations
        """
    )

    # ====================================================================
    # TAB 1: MALNUTRITION DETECTION
    # ====================================================================
    with gr.Tab("🔍 Malnutrition Detection"):
        gr.Markdown("### Upload Image for Analysis")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Child's Facial Image",
                    sources=["upload", "webcam"],
                    interactive=True
                )
                submit_btn = gr.Button("🔍 Analyze Image", size="lg", variant="primary")

            with gr.Column(scale=1):
                prediction_output = gr.Textbox(
                    label="Detection Result",
                    lines=4,
                    interactive=False
                )
                results_output = gr.JSON(
                    label="Detailed Results"
                )

        gr.Markdown("---")

        advisory_output = gr.Textbox(
            label="Health Advisory",
            lines=6,
            interactive=False
        )

        # Connect button to prediction function
        submit_btn.click(
            fn=predict_and_advise,
            inputs=image_input,
            outputs=[prediction_output, advisory_output, results_output]
        )

    # ====================================================================
    # TAB 2: MALNUTRITION Q&A
    # ====================================================================
    with gr.Tab("❓ Ask About Malnutrition"):
        gr.Markdown("### Ask Questions About Malnutrition")
        gr.Markdown(
            """
            Ask any questions about child malnutrition, symptoms, prevention, 
            or treatment. Our AI will provide detailed answers based on medical knowledge.
            """
        )

        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the signs of malnutrition in children?",
                    lines=3,
                    interactive=True
                )
                ask_btn = gr.Button("🤖 Get Answer", size="lg", variant="primary")

            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=8,
                    interactive=False
                )

        # Connect button to QA function
        ask_btn.click(
            fn=answer_question,
            inputs=question_input,
            outputs=answer_output
        )

        # Example questions
        gr.Examples(
            examples=[
                ["What are the signs of child malnutrition?"],
                ["How can malnutrition affect child development?"],
                ["What are the best ways to prevent malnutrition in children?"],
                ["What treatments are available for malnourished children?"],
                ["How does malnutrition affect the immune system?"]
            ],
            inputs=question_input,
            outputs=answer_output,
            fn=answer_question,
            cache_examples=False
        )

    # ====================================================================
    # TAB 3: INFORMATION
    # ====================================================================
    with gr.Tab("ℹ️ Information"):
        gr.Markdown(
            """
            ## About This System

            ### Technology Stack
            - **Detection Model**: ResNet50 (trained on facial malnutrition indicators)
            - **LLM**: Mistral 7B (for health advisories and Q&A)
            - **Framework**: PyTorch + Gradio

            ### How It Works
            1. **Image Analysis**: Analyzes facial features for malnutrition indicators
            2. **Prediction**: Uses deep learning to classify health status
            3. **Advisory**: Generates personalized health recommendations using AI

            ### Accuracy
            - Detection Accuracy: ~92% on validation set
            - Confidence Scores: Probability-based reliability metric

            ### Important Notes
            ⚠️ **Disclaimer**: This system is for screening purposes only and should not 
            replace professional medical diagnosis. Always consult healthcare professionals 
            for proper diagnosis and treatment.

            ### Supported Predictions
            - ✅ **Healthy**: Child appears to be in good nutritional status
            - ⚠️ **Malnourished**: Child shows signs requiring medical attention

            ### Tips for Best Results
            1. Use well-lit, clear facial images
            2. Ensure the face is clearly visible in the image
            3. Avoid heavy shadows or profile angles
            4. Use recent photographs for accurate assessment
            """
        )

    # ====================================================================
    # FOOTER
    # ====================================================================
    gr.Markdown("---")
    gr.Markdown(
        """
        <div style="text-align: center">
        <p><b>Malnutrition Detection System v1.0</b></p>
        <p>Powered by ResNet50 + Mistral 7B</p>
        <p style="font-size: 12px; color: gray;">
        This system is for educational and screening purposes. 
        Always consult qualified healthcare professionals for medical advice.
        </p>
        </div>
        """
    )

# ============================================================================
# 8. LAUNCH APP
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting Gradio Server")
    print("="*70)
    print("\n✓ Web app is running!")
    print("\nAccess the app at: http://127.0.0.1:7860")
    print("\nPress Ctrl+C to stop the server")
    print("\n" + "="*70 + "\n")

    # Launch the interface
    demo.queue(max_size=20)  # Enable queuing with max 20 concurrent users
    
    try:
        # Try to launch on default port first
        demo.launch(server_port=7860, share=True, show_error=True)
    except OSError:
        try:
            # If default port is busy, try alternate port
            demo.launch(server_port=8080, share=True, show_error=True)
        except OSError:
            # If both ports are busy, let Gradio choose an available port
            demo.launch(server_port=None, share=True, show_error=True)
