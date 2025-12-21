import mailbox
import os
import re
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
    
    def handle_data(self, data):
        self.text.append(data)
    
    def get_text(self):
        return ''.join(self.text)

def strip_html(html):
    """Remove HTML tags and return plain text."""
    stripper = HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()

def clean_text(text):
    """Clean up email text by removing extra whitespace and formatting."""
    # Remove style tags and their contents
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove script tags and their contents
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Strip HTML if present
    if '<html' in text.lower() or '<div' in text.lower() or '<table' in text.lower():
        text = strip_html(text)
    
    # Remove CSS-like patterns that might remain
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'body\s*\{.*?\}', '', text, flags=re.DOTALL)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove common email artifacts
    text = re.sub(r'\[image:.*?\]', '', text)
    text = re.sub(r'https?://\S+', '[LINK]', text)
    
    # Remove lines that are mostly CSS/code-like (have lots of colons, semicolons, braces)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that look like code/CSS
        if line.count(':') > 3 or line.count(';') > 3 or line.count('{') > 0:
            continue
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    return text.strip()

def parse_mbox_to_files(mbox_path, output_dir='KeepEmail', max_emails=None):
    """
    Parse an mbox file and save each email as a separate text file.
    
    Args:
        mbox_path: Path to the .mbox file
        output_dir: Directory to save emails (default: KeepEmail)
        max_emails: Maximum number of emails to extract (None = all)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the mbox file
    mbox = mailbox.mbox(mbox_path)
    
    count = 0
    for i, message in enumerate(mbox, 1):
        if max_emails and i > max_emails:
            break
        
        # Extract subject and body
        subject = message.get('subject', 'No Subject')
        
        # Get email body (prefer plain text over HTML)
        body = ""
        if message.is_multipart():
            plain_text = None
            html_text = None
            
            for part in message.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain" and not plain_text:
                    plain_text = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif content_type == "text/html" and not html_text:
                    html_text = part.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            # Prefer plain text, fall back to HTML
            body = plain_text if plain_text else html_text
        else:
            body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        # Clean the text
        body = clean_text(body) if body else ""
        
        # Combine subject and body
        email_text = f"Subject: {subject}\n\n{body}"
        
        # Save to file
        filename = os.path.join(output_dir, str(i))
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(email_text)
        
        count += 1
        if count % 50 == 0:
            print(f"Processed {count} emails...")
    
    print(f"\n✅ Saved {count} emails to {output_dir}/")
    return count

if __name__ == "__main__":
    mbox_file = "./mail.mbox"
    parse_mbox_to_files(mbox_file, output_dir='emails_extracted')