import subprocess
import json
import sys
import os
import argparse
import requests

from constants import DEFAULT_PROMPT, DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_PORT
from helpers import display_result, print_welcome_message, read_multiline_input

class LLMEmailClassifier:
    def __init__(self, model='llama3.2:3b', prompt_file=None, ollama_host=None):
        self.model = model
        
        if ollama_host is None:
            host = os.getenv('OLLAMA_HOST', DEFAULT_OLLAMA_HOST)
            port = os.getenv('OLLAMA_PORT', str(DEFAULT_OLLAMA_PORT))
            ollama_host = f'{host}:{port}'
        self.ollama_host = ollama_host
        
        if prompt_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_file = os.path.join(script_dir, 'prompts/default.txt')
        self.prompt_template = self._load_prompt(prompt_file)
        self._check_ollama()
    
    def classify(self, email_text):
        """
        Classify email into categories.
        
        Returns:
            dict with 'category', 'confidence', and 'reasoning'
        """
        prompt = self.prompt_template.format(email_text=email_text[:2000])

        try:
            response = requests.post(
                f'{self.ollama_host}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_response(data['response'])
        
        except requests.exceptions.Timeout:
            return {'category': 'ERROR', 'confidence': 'LOW', 
                   'reasoning': 'LLM timeout', 'should_delete': False}
        except Exception as e:
            return {'category': 'ERROR', 'confidence': 'LOW',
                   'reasoning': f'Error: {str(e)}', 'should_delete': False}

    def _load_prompt(self, prompt_file):
        """Load prompt template from file."""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"⚠️  Prompt file '{prompt_file}' not found. Using default prompt.")
            return self._default_prompt()
    
    def _default_prompt(self):
        return DEFAULT_PROMPT
    
    def _check_ollama(self):
        """Check if ollama is installed and model is available."""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, check=True)
            if self.model.split(':')[0] not in result.stdout:
                print(f"⚠️  Model {self.model} not found. Pulling...")
                subprocess.run(['ollama', 'pull', self.model], check=True)
        except FileNotFoundError:
            print("❌ Ollama not found.")
            sys.exit(1)
    
    def _parse_response(self, response):
        """Parse the LLM response."""
        lines = response.strip().split('\n')
        result = {
            'category': 'UNKNOWN',
            'confidence': 'LOW',
            'reasoning': '',
            'raw_response': response
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('CATEGORY:'):
                result['category'] = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                result['confidence'] = line.split(':', 1)[1].strip()
            elif line.startswith('REASON:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
        
        # Only delete emails with high confidence
        result['should_delete'] = (
            result['category'] in ['DELETE', 'UNWANTED', 'REJECT', 'SPAM'] and 
            result['confidence'] == 'HIGH'
        )
        
        return result
    
    def classify_batch(self, email_texts):
        """Classify multiple emails."""
        results = []
        for i, email in enumerate(email_texts, 1):
            print(f"Classifying email {i}/{len(email_texts)}...")
            result = self.classify(email)
            results.append(result)
        return results

def run_interactive_classifier(classifier):
    """Run interactive email classification loop."""

    while 1:
        print_welcome_message()
        email_text = read_multiline_input()
        
        if email_text.strip().lower() == 'exit':
            break
        
        result = classifier.classify(email_text)
        display_result(result)

def main():
    parser = argparse.ArgumentParser(description='LLM Email Classifier')
    parser.add_argument('file', nargs='?', help='Email file to classify (optional, interactive if not provided)')
    parser.add_argument('--prompt', '-p', help='Path to prompt file', default=None)
    parser.add_argument('--model', '-m', help='Ollama model to use', default='llama3.2:3b')
    
    args = parser.parse_args()
    
    classifier = LLMEmailClassifier(model=args.model, prompt_file=args.prompt)
    
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                email_text = f.read()
            
            print(f"📧 Classifying: {args.file}\n")
            result = classifier.classify(email_text)
            display_result(result)
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
    else:
        run_interactive_classifier(classifier)

if __name__ == "__main__":
    main()