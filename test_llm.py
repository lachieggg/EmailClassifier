import pytest
import os
from email_classify_llm import LLMEmailClassifier
from email_classify_distilbert import EmailClassifier

# Test data paths
TEST_EMAIL_DELETE = "testdata/email_delete.txt"
TEST_EMAIL_OK = "testdata/email_ok.txt"

def load_email(filepath):
    """Helper to load email from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# ============================================
# LLM Classifier Tests
# ============================================

@pytest.fixture
def llm_classifier():
    """Create LLM classifier instance."""
    return LLMEmailClassifier()

def test_llm_should_delete(llm_classifier):
    """Test LLM correctly identifies email that should be deleted."""
    email = load_email(TEST_EMAIL_DELETE)
    result = llm_classifier.classify(email)
    
    print(f"\nLLM Result for DELETE email:")
    print(f"  Category: {result['category']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Reasoning: {result['reasoning']}")
    print(f"  Should delete: {result['should_delete']}")
    
    assert result['category'] == 'DELETE', \
        f"Expected DELETE, got {result['category']}"
    
    if result['confidence'] == 'HIGH':
        assert result['should_delete'] == True, \
            "DELETE with high confidence should be deleted"

def test_llm_should_keep(llm_classifier):
    """Test LLM correctly identifies email that should be kept."""
    email = load_email(TEST_EMAIL_OK)
    result = llm_classifier.classify(email)
    
    print(f"\nLLM Result for OK email:")
    print(f"  Category: {result['category']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Reasoning: {result['reasoning']}")
    print(f"  Should delete: {result['should_delete']}")
    
    assert result['category'] == 'KEEP', \
        f"Expected NORMAL, got {result['category']}"
    assert result['should_delete'] == False, \
        "Normal email should not be deleted"
