#!/usr/bin/env python3
"""
Quick IG completeness test - verifica convergenza con pochi samples
Uso: python scripts/quick_ig_test.py
"""

import sys
import torch
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from src.models.ensemble import EnsembleModel
from src.training.eval_model import load_model_for_eval
from src.config.paths import get_story_file_path

def main():
    story_format = 'narrativo'
    model_name = 'bert-base-uncased'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("  Quick IG Completeness Test")
    print("="*80)
    print(f"Model: {model_name}, Format: {story_format}, Device: {device}\n")
    
    # Load 2 samples
    print("📖 Loading test samples...")
    test_path = get_story_file_path(story_format, 'test')
    with open(test_path, 'rb') as f:
        texts = pickle.load(f)[:2]
    
    label_test_path = get_story_file_path(story_format, 'label_test')
    with open(label_test_path, 'rb') as f:
        labels_str = pickle.load(f)[:2]
    
    label_train_path = get_story_file_path(story_format, 'label_train')
    with open(label_train_path, 'rb') as f:
        labels_train_str = pickle.load(f)
    
    unique_labels = sorted(set(labels_train_str))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label2id[l] for l in labels_str]
    num_classes = len(label2id)
    
    print(f"   ✅ Loaded {len(texts)} samples\n")
    
    # Load ensemble
    print("🤖 Loading ensemble...")
    model, is_ensemble, hf_model_id = load_model_for_eval(
        story_format=story_format,
        model_name=model_name,
        num_classes=num_classes,
        device=device,
        use_ensemble=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    print(f"   ✅ Loaded {len(model.models)} folds\n")
    
    # Test n_steps values
    n_steps_values = [20, 50, 100]
    
    print(f"🔍 Testing n_steps: {n_steps_values}\n")
    
    for sample_idx, (text, label) in enumerate(zip(texts, labels)):
        print(f"{'='*80}")
        print(f"Sample {sample_idx+1}/{len(texts)}")
        print(f"{'='*80}\n")
        
        # Tokenize
        encoding = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get prediction
        with torch.no_grad():
            probs = model.predict(input_ids, attention_mask)
            predicted_class = torch.argmax(probs, dim=1).item()
        
        print(f"Text preview: {text[:100]}...")
        print(f"True label: {label}, Predicted: {predicted_class}\n")
        
        for n_steps in n_steps_values:
            print(f"Testing n_steps={n_steps}:")
            
            # Compute with completeness check
            try:
                attributions = model.compute_ensemble_attributions(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_class=predicted_class,
                    n_steps=n_steps,
                    check_completeness=True,
                    auto_increase_steps=False,
                    verbose=True
                )
                print(f"   ✅ Attributions computed: shape={attributions.shape}\n")
            except Exception as e:
                print(f"   ❌ Error: {e}\n")
    
    print(f"\n{'='*80}")
    print("✅ Test completato!")
    print(f"{'='*80}\n")
    
    print("💡 Raccomandazione:")
    print("   - Se rel_error < 0.01 (1%) con n_steps=50 → OK")
    print("   - Se rel_error > 0.05 (5%) → aumentare a 100 o 200")
    print("   - Convergenza ottimale: rel_error < 0.01, convergence_rate > 95%")
    print()

if __name__ == '__main__':
    main()
