import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        words = [self.pad_token, self.unk_token, self.bos_token, self.eos_token] 
        
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
            unique_words = sorted(set(all_words))

        words.extend(unique_words)
            
        idxs = range(0, len(words))
        self.word_to_id = dict(zip(words, idxs))
        self.id_to_word = dict(zip(idxs, words))
        self.vocab_size = len(words)
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        keys = self.word_to_id.keys()
        arr = text.lower().split()
        
        ids = [self.word_to_id[i] if i in keys else self.word_to_id[self.unk_token] for i in arr]
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        
        keys = self.id_to_word.keys()
        words = [self.id_to_word[i] if i in keys else self.unk_token for i in ids]
        
        return ' '.join(words)
