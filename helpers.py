from sys import exit

def display_result(result):
    """Display classification result in formatted output."""
    print(f"\nCategory: {result['category']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Action: {'❌ DELETE' if result['should_delete'] else '✅ KEEP'}")

def print_welcome_message():
    """Print welcome message."""
    print("Enter email and press Ctrl+D (Mac, Linux) or Ctrl+Z (Windows) then Enter\n")

def read_multiline_input():
    """Read multi-line input until EOF (Ctrl+D/Ctrl+Z)."""
    lines = []
    try:
        while 1:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    except KeyboardInterrupt:
        print("\nExiting.")
        exit(0)
    return '\n'.join(lines)