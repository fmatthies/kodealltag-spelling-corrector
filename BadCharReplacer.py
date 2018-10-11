# -*- coding: utf-8 -*-


class BadCharReplacer(object):
    def __init__(self, word_dict_obj):
        self._umlaute = "äöüß"
        self._alphabet = "abcdefghijklmnopqrstuvwxyz" + self._umlaute
        self._repl_chr = u"\uFFFD" # equals the following: �
        self._word_dict = word_dict_obj
        self._bchr_count = 0
    
    def _edits1(self, word, uml_only=True):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
        if uml_only:
            replaces = [a + c + b[1:] for a, b in splits for c in self._umlaute if b]
            inserts = [a + c + b for a, b in splits for c in self._umlaute]
        else:
            replaces = [a + c + b[1:] for a, b in splits for c in self._alphabet if b]
            inserts = [a + c + b for a, b in splits for c in self._alphabet]
        return set(deletes + transposes + replaces + inserts)
    
    def _known_edits2(self, word, uml_only=True):
        return set(e2 for e1 in self._edits1(word, uml_only) \
                   for e2 in self._edits1(e1, uml_only) \
                   if e2 in self._word_dict.get_word_dict())
    
    def _known(self, words):
        return set(w for w in words if w in self._word_dict.get_word_dict())
    
    def _correct(self, word, uml_only=True):
        candidates = self._known([word]) or self._known(self._edits1(word, uml_only)) \
            or self._known_edits2(word, uml_only) or [word]
        return max(candidates, key=self._word_dict.get_word_dict().get)
    
    def _unigram_replacement(self, word, uml_only=True):
        return word
    
    def clear_chr_count(self):
        self._bchr_count = 0
        
    def get_chr_count(self):
        return self._bchr_count
        
    def repl_unicode_qmark(self, word, uml_only=True):
        if word.count(self._repl_chr) == 0:
            return word
        elif word.count(self._repl_chr) <= 2:
            self._bchr_count += 1
            return self._correct(word, uml_only)
        else:
            self._bchr_count += 1
            return self._unigram_replacement(word, uml_only)
