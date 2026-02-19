import os
import random
import json
from langchain_core.tools import tool

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
    
    