import torch
import sys
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class EmailClassifier:
    def __init__(self, model_path='./email_filter_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def classify(self, email_text, return_confidence=False):
        """
        Classify an email as normal (0) or unwanted (1).
        
        Args:
            email_text: The email content as a string
            return_confidence: If True, return confidence scores
        
        Returns:
            If return_confidence=False: 0 (normal) or 1 (unwanted)
            If return_confidence=True: (prediction, confidence_score)
        """
        # Tokenize
        encoding = self.tokenizer(
            email_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        if return_confidence:
            return prediction, confidence
        return prediction
    
    def classify_batch(self, email_texts):
        """Classify multiple emails at once."""
        results = []
        for email_text in email_texts:
            pred, conf = self.classify(email_text, return_confidence=True)
            results.append({
                'prediction': 'Unwanted' if pred == 1 else 'Normal',
                'confidence': conf,
                'should_delete': pred == 1 and conf > 0.85
            })
        return results

if __name__ == "__main__":
    classifier = EmailClassifier('./email_filter_model')
    
    while 1:
        print("\nPaste email content (press Ctrl+D when done, or type 'exit' to quit):")
        email_text = sys.stdin.read()
        if email_text.strip().lower() == 'exit':
            break
        prediction, confidence = classifier.classify(email, return_confidence=True)
        label = "Unwanted" if prediction == 1 else "Normal"
        action = "❌ DELETE" if prediction == 1 and confidence > 0.85 else "✅ KEEP"
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Action: {action}")