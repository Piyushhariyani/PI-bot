from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Solve mathematical expressions"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"


@tool
def word_counter(text: str) -> str:
    """Counts words in a sentence"""
    return f"Total words: {len(text.split())}"