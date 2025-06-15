from langchain_google_genai import ChatGoogleGenerativeAI
import os

def translate_text(text: str) -> str:
    """
    Translates text using Google's Gemini LLM based on user's natural language request.
    
    The function accepts any translation request in natural language format and uses
    the LLM to understand and perform the translation accordingly.
    
    Args:
        text (str): The input text containing both the translation request and text to translate.
                   Examples: 
                   - "Translate 'Hello world' to Spanish"
                   - "Convert 'Bonjour' from French to English"
                   - "What is 'Hola' in English?"
    
    Returns:
        str: The translated text as understood and processed by the LLM
        
    Raises:
        ValueError: If the input text is empty or None
        Exception: If translation fails due to API errors
        
    Examples:
        >>> translate_text("Translate 'Hello world' to Spanish")
        "Hola mundo"
        
        >>> translate_text("Convert 'Bonjour le monde' from French to English")
        "Hello world"
        
        >>> translate_text("What does 'Guten Tag' mean in English?")
        "Good day"
    """
    
    # Input validation
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty or None")
    
    try:
        # Initialize the Gemini LLM using LangChain
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create the flexible translation prompt
        prompt = f"""You are a professional translator. Process the following user request and provide only the translation result without any additional text, explanations, or formatting.

        User request: {text}

        Instructions:
        - Understand what the user wants to translate and to which language
        - If source language is not specified, detect it automatically
        - If target language is not specified, default to English
        - Provide only the translated text as output
        - Handle various request formats like "translate X to Y", "what is X in Y", "convert X from Y to Z", etc.
        """
        
        # Generate the translation
        response = llm.invoke(prompt)
        
        # Extract and return the translated text
        translated_text = response.content.strip()
        return translated_text
        
    except Exception as e:
        raise Exception(f"Translation failed: {str(e)}")

# Example usage:
# result = translate_text("Translate 'Hello world' to Spanish")
# print(result)  # Output: "Hola mundo"

result = translate_text("What does 'Bonjour' mean in English?")
print(result)  # Output: "Hello"

# result = translate_text("Convert 'Guten Tag' from German to French")
# print(result)  # Output: "Bonjour"