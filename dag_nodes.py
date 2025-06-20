"""
LangGraph DAG nodes for self-healing classification
"""
import logging
from typing import Dict, Any, TypedDict, Optional
from model_wrapper import SentimentClassifier, BackupClassifier
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """State structure for the DAG"""
    text: str
    predicted_label: Optional[str]
    confidence: Optional[float]
    full_results: Optional[Dict]
    needs_fallback: bool
    fallback_activated: bool
    user_feedback: Optional[str]
    final_label: Optional[str]
    method_used: str
    timestamp: str
    confidence_threshold: float

class InferenceNode:
    """Node for running initial model inference"""
    
    def __init__(self, model_path: str):
        self.classifier = SentimentClassifier(model_path)
        
    def __call__(self, state: GraphState) -> GraphState:
        """Run inference on the input text"""
        logger.info(f"[InferenceNode] Processing: {state['text'][:50]}...")
        
        try:
            predicted_label, confidence, full_results = self.classifier.predict(state['text'])
            
            # Log the prediction
            logger.info(f"[InferenceNode] Predicted label: {predicted_label} | Confidence: {confidence:.1%}")
            
            # Update state
            state.update({
                'predicted_label': predicted_label,
                'confidence': confidence,
                'full_results': full_results,
                'method_used': 'fine_tuned_model',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"[InferenceNode] Error during inference: {e}")
            # Set fallback flag
            state.update({
                'needs_fallback': True,
                'method_used': 'error_fallback'
            })
            
        return state

class ConfidenceCheckNode:
    """Node for evaluating prediction confidence"""
    
    def __call__(self, state: GraphState) -> GraphState:
        """Check if confidence is above threshold"""
        confidence = state.get('confidence', 0.0)
        threshold = state.get('confidence_threshold', 0.7)
        
        if confidence < threshold:
            logger.info(f"[ConfidenceCheckNode] Confidence too low ({confidence:.1%} < {threshold:.1%}). Triggering fallback...")
            state['needs_fallback'] = True
        else:
            logger.info(f"[ConfidenceCheckNode] Confidence acceptable ({confidence:.1%} >= {threshold:.1%})")
            state['needs_fallback'] = False
            state['final_label'] = state['predicted_label']
            
        return state

class FallbackNode:
    """Node for handling fallback scenarios"""
    
    def __init__(self, model_path: str = None):
        self.backup_classifier = BackupClassifier()
        self.primary_classifier = None
        if model_path:
            try:
                self.primary_classifier = SentimentClassifier(model_path)
            except Exception as e:
                logger.warning(f"Could not load primary classifier for fallback: {e}")
        
    def __call__(self, state: GraphState) -> GraphState:
        """Handle fallback logic"""
        logger.info("[FallbackNode] Fallback activated")
        
        state['fallback_activated'] = True
        
        # Strategy 1: Ask user for clarification
        user_input = self._ask_user_clarification(state)
        
        if user_input and user_input.lower() not in ['skip', 'backup']:
            # Process user feedback
            final_label = self._process_user_feedback(user_input, state)
            state.update({
                'user_feedback': user_input,
                'final_label': final_label,
                'method_used': 'user_clarification'
            })
            logger.info(f"[FallbackNode] Final label from user: {final_label}")
            
        else:
            # Strategy 2: Use backup classifier
            logger.info("[FallbackNode] Using backup classifier")
            backup_label, backup_confidence, backup_results = self.backup_classifier.predict(state['text'])
            
            state.update({
                'final_label': backup_label,
                'confidence': backup_confidence,
                'full_results': backup_results,
                'method_used': 'backup_classifier'
            })
            logger.info(f"[FallbackNode] Backup prediction: {backup_label} | Confidence: {backup_confidence:.1%}")
            
        return state
        
    def _ask_user_clarification(self, state: GraphState) -> str:
        """Ask user for clarification"""
        text = state['text']
        predicted_label = state.get('predicted_label', 'Unknown')
        confidence = state.get('confidence', 0.0)
        
        print(f"\n{'='*60}")
        print(f"🤔 CLARIFICATION NEEDED")
        print(f"{'='*60}")
        print(f"Text: {text}")
        print(f"Initial prediction: {predicted_label} (Confidence: {confidence:.1%})")
        print(f"\nThe model is unsure about this prediction.")
        print(f"Could you help clarify the sentiment?")
        print(f"\nOptions:")
        print(f"1. Type 'positive' if this is positive sentiment")
        print(f"2. Type 'negative' if this is negative sentiment")
        print(f"3. Type 'backup' to use backup classifier")
        print(f"4. Type 'skip' to skip user input")
        print(f"{'='*60}")
        
        try:
            user_input = input("Your input: ").strip()
            return user_input
        except (EOFError, KeyboardInterrupt):
            return "skip"
            
    def _process_user_feedback(self, user_input: str, state: GraphState) -> str:
        """Process user feedback to determine final label"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['positive', 'good', 'yes', 'correct']):
            return "POSITIVE"
        elif any(word in user_input_lower for word in ['negative', 'bad', 'no', 'wrong']):
            return "NEGATIVE"
        else:
            # Try to infer from context
            if "was" in user_input_lower and any(word in user_input_lower for word in ['not', 'negative', 'bad']):
                return "NEGATIVE"
            elif "was" in user_input_lower and any(word in user_input_lower for word in ['positive', 'good']):
                return "POSITIVE"
            else:
                # Default to original prediction if unclear
                return state.get('predicted_label', 'UNKNOWN')

class LoggingNode:
    """Node for structured logging"""
    
    def __init__(self, log_file: str = "classification_log.json"):
        self.log_file = log_file
        
    def __call__(self, state: GraphState) -> GraphState:
        """Log the final state"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_text": state['text'],
            "initial_prediction": state.get('predicted_label'),
            "initial_confidence": state.get('confidence'),
            "confidence_threshold": state.get('confidence_threshold'),
            "fallback_activated": state.get('fallback_activated', False),
            "user_feedback": state.get('user_feedback'),
            "final_label": state.get('final_label'),
            "method_used": state.get('method_used'),
            "full_results": state.get('full_results')
        }
        
        # Append to log file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
            
        # Console logging
        print(f"\n📊 FINAL RESULT")
        print(f"{'='*40}")
        print(f"Input: {state['text']}")
        print(f"Final Label: {state.get('final_label', 'Unknown')}")
        print(f"Method: {state.get('method_used', 'Unknown')}")
        if state.get('confidence'):
            print(f"Confidence: {state['confidence']:.1%}")
        if state.get('fallback_activated'):
            print(f"Fallback: Activated")
        print(f"{'='*40}\n")
        
        return state

def route_after_confidence_check(state: GraphState) -> str:
    """Routing function to determine next node after confidence check"""
    if state.get('needs_fallback', False):
        return "fallback"
    else:
        return "logging"

def should_continue(state: GraphState) -> str:
    """Determine if workflow should continue"""
    if state.get('final_label'):
        return "logging"
    else:
        return "fallback"