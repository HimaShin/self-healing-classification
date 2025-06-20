# Self-Healing Classification DAG with Fine-Tuned Model

A robust sentiment classification system built with LangGraph that incorporates self-healing mechanisms and human-in-the-loop workflows for high-confidence predictions.

## 🚀 Features

- **Fine-tuned Transformer**: DistilBERT fine-tuned with LoRA on IMDB sentiment dataset
- **Self-Healing DAG**: LangGraph-based workflow with confidence-based fallback
- **Human-in-the-Loop**: Interactive clarification when confidence is low
- **Backup Classifier**: Rule-based fallback for system resilience
- **Structured Logging**: Comprehensive logging of predictions and fallbacks
- **Interactive CLI**: User-friendly command-line interface

## 🏗️ Architecture

```
Input Text
    ↓
InferenceNode (Fine-tuned Model)
    ↓
ConfidenceCheckNode
    ↓
[High Confidence] → LoggingNode → End
    ↓
[Low Confidence] → FallbackNode → LoggingNode → End
                        ↓
            [User Clarification OR Backup Model]
```

## 📋 Requirements

- Python 3.10.11
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- Internet connection for dataset download

## 🛠️ Installation

1. **Clone or create the project directory**:
```bash
mkdir self-healing-classification
cd self-healing-classification
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Step 1: Fine-tune the Model

Train the sentiment classification model on IMDB dataset:

```bash
python finetune_model.py
```

This will:
- Download the IMDB dataset
- Fine-tune DistilBERT with LoRA
- Save the model to `./fine_tuned_model/`
- Training takes ~20-30 minutes on GPU

**Training Output Example**:
```
Loading IMDB dataset...
Dataset loaded: 5000 train, 1000 test samples
Loading model and tokenizer: distilbert-base-uncased
Setting up LoRA configuration...
trainable params: 1,327,106 || all params: 68,119,554 || trainable%: 1.9473
Starting training...
Epoch 1/3: 100%|██████████| 313/313 [08:42<00:00, 1.67s/it]
Epoch 2/3: 100%|██████████| 313/313 [08:38<00:00, 1.66s/it]
Epoch 3/3: 100%|██████████| 313/313 [08:41<00:00, 1.66s/it]
Training completed. Model saved to ./fine_tuned_model
Final results: {'eval_accuracy': 0.896, 'eval_f1': 0.896}
```

### Step 2: Run the Classification System

#### Interactive Mode (Recommended)
```bash
python main.py
```

#### Single Text Classification
```bash
python main.py --text "The movie was absolutely terrible and boring"
```

#### Batch Processing from File
```bash
python main.py --batch-file texts.txt
```

#### Custom Confidence Threshold
```bash
python main.py --confidence-threshold 0.8
```

## 💬 CLI Usage

### Interactive Commands

- `classify <text>` - Classify sentiment of text
- `batch` - Enter batch processing mode  
- `stats` - Show classification statistics
- `workflow` - Display DAG structure
- `help` - Show help message
- `quit` - Exit system

### Example Session

```
🎯 SELF-HEALING SENTIMENT CLASSIFICATION SYSTEM
   Built with LangGraph & Fine-tuned Transformer
==================================================================

> classify The movie was painfully slow and boring

🔍 Processing: The movie was painfully slow and boring
============================================================
[InferenceNode] Predicted label: POSITIVE | Confidence: 54.2%
[ConfidenceCheckNode] Confidence too low (54.2% < 70.0%). Triggering fallback...
[FallbackNode] Fallback activated

============================================================
🤔 CLARIFICATION NEEDED
============================================================
Text: The movie was painfully slow and boring
Initial prediction: POSITIVE (Confidence: 54.2%)

The model is unsure about this prediction.
Could you help clarify the sentiment?

Options:
1. Type 'positive' if this is positive sentiment
2. Type 'negative' if this is negative sentiment  
3. Type 'backup' to use backup classifier
4. Type 'skip' to skip user input
============================================================
Your input: Yes, it was definitely negative

[FallbackNode] Final label from user: NEGATIVE

📊 FINAL RESULT
========================================
Input: The movie was painfully slow and boring
Final Label: NEGATIVE
Method: user_clarification
Fallback: Activated
========================================
```

## 📊 Expected Workflow Output

The system demonstrates the self-healing capability:

1. **High Confidence**: Direct classification
```
[InferenceNode] Predicted label: NEGATIVE | Confidence: 95.3%
[ConfidenceCheckNode] Confidence acceptable (95.3% >= 70.0%)
Final Label: NEGATIVE
```

2. **Low Confidence**: Fallback activation
```
[InferenceNode] Predicted label: POSITIVE | Confidence: 54%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?
User: Yes, it was definitely negative.
Final Label: NEGATIVE (Corrected via user clarification)
```

## 📁 Project Structure

```
self-healing-classification/
├── requirements.txt          # Python dependencies
├── finetune_model.py        # Model training script
├── model_wrapper.py         # Model inference wrapper
├── dag_nodes.py             # LangGraph DAG nodes
├── dag_workflow.py          # DAG workflow definition
├── main.py                  # Main CLI interface
├── README.md               # This file
├── fine_tuned_model/       # Trained model directory
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── tokenizer_config.json
│   └── training_results.json
├── classification_log.json # Structured logs
└── classification_system.log # System logs
```

## 🔧 Configuration

### Confidence Threshold
- Default: 70%
- Adjustable via `--confidence-threshold` parameter
- Higher values = more fallbacks = higher accuracy

### Model Parameters
- Base model: DistilBERT
- LoRA rank: 16
- Training epochs: 3
- Batch size: 16

### Fallback Strategies
1. **User Clarification**: Interactive human feedback
2. **Backup Classifier**: Rule-based sentiment analysis

## 📈 Logging and Monitoring

### Structured Logs (`classification_log.json`)
Each classification generates a JSON log entry:
```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "input_text": "The movie was great!",
  "initial_prediction": "POSITIVE",
  "initial_confidence": 0.953,
  "confidence_threshold": 0.7,
  "fallback_activated": false,
  "user_feedback": null,
  "final_label": "POSITIVE",
  "method_used": "fine_tuned_model"
}
```

### Statistics View
Use `stats` command to see:
- Total classifications
- Fallback activation rate
- Method distribution
- Label distribution
- Average confidence scores

## 🎬 Demo Video

The demo video should showcase:

1. **Model Training**: Running `finetune_model.py`
2. **CLI Launch**: Starting the interactive system
3. **High Confidence Case**: Direct classification
4. **Low Confidence Case**: Fallback activation and user clarification
5. **Batch Processing**: Multiple text classification
6. **Statistics**: Viewing classification stats
7. **DAG Explanation**: Workflow structure explanation

## 🚨 Troubleshooting

### Common Issues

1. **Model not found error**:
   ```bash
   # Make sure to train the model first
   python finetune_model.py
   ```

2. **CUDA out of memory**:
   ```bash
   # Reduce batch size in finetune_model.py
   per_device_train_batch_size=8  # Instead of 16
   ```

3. **Dependencies issues**:
   ```bash
   pip install --upgrade torch transformers
   ```

4. **Permission errors on Windows**:
   ```bash
   # Run as administrator or use conda environment
   ```

## 🔄 Extending the System

### Adding New Models
1. Modify `model_wrapper.py` to support new architectures
2. Update `finetune_model.py` for different datasets
3. Adjust node logic in `dag_nodes.py`

### Custom Fallback Strategies
1. Extend `FallbackNode` class
2. Add new routing logic in `dag_workflow.py`
3. Update CLI interface for new options

### Advanced Logging
1. Add database logging in `LoggingNode`
2. Implement real-time monitoring dashboard
3. Add email alerts for high fallback rates

## 📝 Performance Metrics

Expected performance on IMDB dataset:
- **Accuracy**: ~89-92%
- **F1 Score**: ~89-92%
- **Training Time**: 20-30 minutes (GPU)
- **Inference Speed**: ~50ms per text
- **Model Size**: ~250MB (with LoRA)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## 📄 License

This project is for educational and evaluation purposes. Please ensure compliance with dataset licenses and model terms of use.

---

