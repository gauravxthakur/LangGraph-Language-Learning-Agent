import os
import random
import json
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import List
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

load_dotenv()

translation_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool

def get_n_random_words(language: str, n: int) -> list:
    
    """
    Retrieve a specified number of random words from a language-specific word list.
    
    Args:
        language (str): The language code (e.g., 'spanish', 'french') to determine 
                       which word list to use. Must correspond to a directory 
                       in the 'data' folder.
        n (int): The number of random words to retrieve. Must be less than or 
                equal to the total number of words in the word list.
    
    Returns:
        list: A list of randomly selected words from the specified language's 
              word list. Each word is returned as a string.
    
    Raises:
        FileNotFoundError: If the word list file doesn't exist for the specified 
                          language.
        KeyError: If the word list file structure is invalid.
        ValueError: If n is larger than the available words in the list.
    
    Note:
        The function expects word lists to be stored in JSON format at:
        'data/{language}/word-list-cleaned.json'
        Each entry in the JSON should have a 'word' key containing the word.
    """
    
    path = os.path.join("data",f"{language}", "word-list-cleaned.json")
    
    with open(path, "r") as f:
        word_list = json.load(f)
        
    random_word_dict = {k: word_list[k] for k in random.sample(list(word_list.keys()), n)}
    random_words = [item["word"] for item in random_word_dict.values()]
    
    return random_words

@tool
def get_n_random_words_by_difficulty_level(language:str,
                                           difficulty_level:str,
                                           n:int) -> list:
    """
    Retrieve a specified number of random words from a language-specific word list,
    filtered by difficulty level.
 
    Args:
        language (str): The language code (e.g., 'spanish', 'french') to determine 
                       which word list to use. Must correspond to a directory 
                       in the 'data' folder.
        difficulty_level (str): The difficulty level to filter words by. Must match
                               the 'difficulty_level' field in the word list entries.
                               The only valid values are "beginner", "intermediate", "advanced".
        n (int): The number of random words to retrieve. Must be less than or 
                equal to the total number of words available for the specified 
                difficulty level.
 
    Returns:
        list: A list of randomly selected words from the specified language's 
              word list that match the given difficulty level. Each word is 
              returned as a string.
 
    Raises:
        FileNotFoundError: If the word list file doesn't exist for the specified 
                          language.
        KeyError: If the word list file structure is invalid or if 'difficulty_level'
                 field is missing from word entries.
        ValueError: If n is larger than the available words for the specified 
                   difficulty level, or if no words match the difficulty level.
 
    Note:
        The function expects word lists to be stored in JSON format at:
        'data/{language}/word-list-cleaned.json'
        Each entry in the JSON should have a 'word' key containing the word and 
        a 'word_difficulty' key containing the difficulty classification.
    """
    path = os.path.join("data",f"{language}", "word-list-cleaned.json")
    
    with open(path, "r") as f:
        word_list = json.load(f)
        
    words_filtered_by_difficulty = {k: v for k, v in word_list.items() if v["word_difficulty"] == difficulty_level}
    
    random_word_dict = {k: words_filtered_by_difficulty[k] for k in random.sample(list(words_filtered_by_difficulty.keys()), n)}
    random_words = [item["word"] for item in random_word_dict.values()]
    
    return random_words
    
    
    
@tool
def translate_words(random_words: List[str],
                    source_language: str,
                    target_language: str) -> dict:
    
    """
    Translate a list of words from source to target language using a llm.
    
    Args:
        random_words (list): List of words to translate.
        source_language (str): Source language name.
        target_language (str): Target language name.
    
    Returns:
        dict: {"translations": [{"source": word, "target": translation}, ...]}
    
    Raises:
        ValueError: If AI response cannot be parsed as valid JSON.
    """
    prompt = (
        f"You are a precise translation engine.\n"
        f"Translate the following {len(random_words)} words from {source_language} to {target_language}.\n"
        f"Return ONLY valid JSON with this exact structure.\n"
        f'{{"translations": [{{"source":"<original>","target":"<translated>"}}]}}\n'
        f"No explanations, no extra fields, no markdown, no comments.\n"
        f"Words: {json.dumps(random_words, ensure_ascii=False)}"
    )
    
    response = translation_model.invoke([HumanMessage(content=prompt)])
    text = getattr(response, "content", str(response))
    
    # Try to parse JSON strictly: if it fails, attempt to extract the first JSON object.
    
    try:
        parsed = json.loads(text)
    except Exception:
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            raise ValueError("Could not parse JSON from translation response.")
        
    translations_list = parsed.get("translations", [])
    # Build a mapping from the model output
    model_map = {item.get("source", ""): item.get("target", "") for item in translations_list if isinstance(item, dict)}
    
    # Ensure we return translations in the same order as input: fall back to identity if missing
    ordered_translations = [
        {"source": w, "target": model_map.get(w, model_map.get(w.capitalize(), w))} for w in random_words
    ]
    return {
        "translations": ordered_translations
    }