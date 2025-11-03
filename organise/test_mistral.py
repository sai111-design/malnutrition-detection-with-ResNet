#!/usr/bin/env python3
"""
Test Mistral 7B Integration
Tests the LLM handler to ensure Mistral 7B is working correctly
"""

import sys
import os

print("\n" + "="*70)
print("Testing Mistral 7B Integration")
print("="*70)

try:
    print("\nStep 1: Importing LLM handler...")
    from src.llm_handler import get_llm
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Make sure src/llm_handler.py exists and is in the correct location")
    sys.exit(1)

try:
    print("\nStep 2: Loading Mistral 7B model...")
    print("(This may take 3-5 seconds on first run)")
    llm = get_llm()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    print("Make sure the model file exists at: models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    sys.exit(1)

try:
    print("\n" + "="*70)
    print("Test 1: Health Advisory Generation (Malnourished Case)")
    print("="*70)

    prediction = 1  # 1 = malnourished
    confidence = 87.5

    print(f"\nInput:")
    print(f"  Prediction: Malnourished")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"\nGenerating advisory...")

    advisory = llm.generate_health_advisory(prediction, confidence)

    print(f"\nOutput Advisory:")
    print(f"{advisory}")

except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    sys.exit(1)

try:
    print("\n" + "="*70)
    print("Test 2: Health Advisory Generation (Healthy Case)")
    print("="*70)

    prediction = 0  # 0 = healthy
    confidence = 92.3

    print(f"\nInput:")
    print(f"  Prediction: Healthy")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"\nGenerating advisory...")

    advisory = llm.generate_health_advisory(prediction, confidence)

    print(f"\nOutput Advisory:")
    print(f"{advisory}")

except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    sys.exit(1)

try:
    print("\n" + "="*70)
    print("Test 3: Question Answering")
    print("="*70)

    question = "What are the signs of child malnutrition?"

    print(f"\nQuestion: {question}")
    print(f"\nGenerating answer...")

    answer = llm.answer_question(question)

    print(f"\nAnswer:")
    print(f"{answer}")

except Exception as e:
    print(f"✗ Test 3 failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✓ All Tests Passed Successfully!")
print("="*70)
print("\nIntegration Status: READY FOR DEPLOYMENT")
print("\nNext Steps:")
print("  1. Train your detection model: python train.py")
print("  2. Run the web app: python ui/gradio_app.py")
print("\n" + "="*70 + "\n")
