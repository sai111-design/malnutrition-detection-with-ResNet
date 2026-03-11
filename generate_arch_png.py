import base64
import requests
import json
import os

mermaid_code = """flowchart TD
    User([User / Healthcare Worker])
    UI[Gradio Web Application Interface]
    
    subgraph InputProcessing [Input Layer]
        ImgInput[Image Upload / Webcam Capture]
        TextQuery[Educational Q&A Query]
        Preprocess[Preprocessing Resize 224x224, Normalize to Tensor]
    end

    subgraph AIProcessing [AI Processing Engines]
        direction LR
        VisionEngine[Vision Engine ResNet-50 Classifier]
        LLMEngine[LLM Engine Mistral 7B Instruct GGUF]
    end

    subgraph OutputLayer [Aggregation & Outputs]
        DetectResult[Detection Output Class Label & Confidence %]
        Advisory[AI Health Advisory]
        QAResponse[Educational Response]
    end

    User <-->|Interacts with tabs| UI
    
    UI -->|Image Provided| ImgInput
    UI -->|Question Asked| TextQuery
    
    ImgInput --> Preprocess
    Preprocess --> VisionEngine
    
    VisionEngine --> |1. Generates| DetectResult
    DetectResult --> |2. Injects Context into Prompt| LLMEngine
    TextQuery --> |Direct Inference| LLMEngine
    
    LLMEngine --> |Synthesizes| Advisory
    LLMEngine --> |Generates| QAResponse
    
    DetectResult -->|Displays on UI| UI
    Advisory -->|Displays on UI| UI
    QAResponse -->|Displays on UI| UI
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef userNode fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,color:#000;
    classDef uiNode fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000;
    classDef inputNode fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef engineNode fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000;
    classDef outputNode fill:#e0f7fa,stroke:#006064,stroke-width:2px,color:#000;
    
    class User userNode;
    class UI uiNode;
    class ImgInput,TextQuery,Preprocess inputNode;
    class VisionEngine,LLMEngine engineNode;
    class DetectResult,Advisory,QAResponse outputNode;
"""

json_obj = {"code": mermaid_code, "mermaid": {"theme": "default"}}
json_str = json.dumps(json_obj)
b64_str = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')

url = f"https://mermaid.ink/img/{b64_str}"

try:
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs("outputs/images", exist_ok=True)
        with open("outputs/images/system_architecture.png", "wb") as f:
            f.write(response.content)
        print("✓ Architecture diagram saved as PNG successfully.")
    else:
        print(f"✗ Failed to generate image. API Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"✗ Exception occurred: {e}")
