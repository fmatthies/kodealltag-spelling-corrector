# -*- coding: utf-8 -*-

import os
import glob
import dill as pickle # for being able to pickle functions
import collections

class WordDictObject(object):
    
    def __init__(self, word_dict_path):
        self._word_dict_path = os.path.abspath(word_dict_path)
        self._word_dict = collections.defaultdict(lambda: 1)
        
        self._read_word_dict()
        
    def _read_word_dict(self):
        wdp = self.getWordDictPath()
        wd = self.getWordDict()
        
        for pi in glob.glob(os.path.join(wdp, "*.pickle")):
            temp_dict = pickle.load(open(pi, "rb"))
            wd.update(temp_dict)
            
    def getWordDictPath(self):
        return self._word_dict_path
    
    def getWordDict(self):
        return self._word_dict
