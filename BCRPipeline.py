# -*- coding: utf-8 -*-

import treetaggerwrapper
import nltk
import BadCharReplacer
import WordDictObject
import os
import sys
import re
import subprocess
import threading


class EmailWalker(object):
    def __init__(self):
        self.emails = list()
        
    def walk(self, bdir, end=".txt"):
        """ walks a directory, and executes a callback on each file """
        bdir = os.path.abspath(bdir)
        for fi in [fi for fi in os.listdir(bdir) if not fi in [".", ".."]]:
            nfile = os.path.join(bdir, fi)
            if os.path.isdir(nfile):
                self.walk(nfile)
            elif nfile.endswith(end):
                self.emails.append(nfile)
    
    def getEmails(self):
        return self.emails

class PipeThread(threading.Thread):
    def __init__(self, threadID, name, subset, pretokenizer, charrepl, stok, bow=False):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.subset = subset
        self.pretok = pretokenizer
        self.charrepl = charrepl
        self.sent_tokenizer = stok
        self.writeBoW = bow

    def run(self):
        print("Starting {} with {} E-Mails".format(self.name, len(self.subset)))
        for mail in self.subset:
            tagged = list()
            with open(mail) as m:
                txt = m.readlines()
            # returns a list of sentences
            txt = removeComments(txt)
            sents = splitSentences(txt, self.sent_tokenizer)
            for sent in sents:
                # returns a list of tokens
                ctoks = cleanTokens(sent, self.pretok, self.charrepl)
                # returns list of tab seperated items
                tagged.append("\n".join(runTreeTagger(ctoks)))
            self.writeTags(tagged, mail)
        print("Exiting {}".format(self.name))
        
    def writeTags(self, tagged, mail):
        with open(mail+".tok", "w") as tfile:
            tfile.write("\n\n".join(tagged))
        if self.writeBoW:
            with open(mail+".lem", "w") as lfile:
                lfile.write(" ".join(
                    [(line.split("\t")[-1]).lower() for sent in tagged
                        for line in sent.split("\n") if not re.search("(\$.|\$\()", line.split("\t")[1])]
                    )
                )

def getEmails(base_path):
    base_path = os.path.abspath(base_path)
    emails = EmailWalker()
    emails.walk(base_path)
    return emails.getEmails()
        

def getTestText():
    txt_list = \
    ["Ein St�renfried ist der, der unn�tzes Zeug tut und obendrein auch noch st�rt...",
    "Das k�nnte man sicher auch anders beschreiben,so dass wir nicht genau wissen m�gen, in welcher Stra\xDFe dieser eben beschriebene Mensch wohnt.",
    "M�glicherweise ist das aber auch zu viel verlangt und es r�cht sich fr�her oder sp�ter ganz bestimmt.",
    "Wei� jemand genaueres?"]
    return txt_list

def removeComments(txt):
    count = 0
    n_txt = list()
    for line in txt:
        # remove comments and signatures
        if line.startswith("--"):
            break
        # remove first line if intro to quotation
        if count == 0:
            if not re.search("(said|sagte|wrote|schrieb|skrev):", line):
                n_txt.append(line)
        else:
            if not (line.startswith('>') or line.startswith(':')):
                n_txt.append(line)
        count += 1
    return "".join(n_txt)

def cleanTokens(text, pretokenizer, charrepl):
    tokens = pretokenizer.tokenize(text)
    cleaned_tokens = [charrepl.repl_unicode_qmark(token) for token in tokens]
    
    return cleaned_tokens

def splitSentences(txt, stok):
    sents = stok.tokenize(txt)
    return [s.rstrip("\n") for s in sents]

def runTreeTagger(tlist, lang="de"):
    ttagger = treetaggerwrapper.TreeTagger(TAGLANG=lang)
    
    # tags is a list of tab seperated items:
    #   [WORD1\tPOS1\tLEMMA1, WORD2\tPOS2\tLEMMA2, ...]
    tags = ttagger.tag_text(" ".join(tlist))
    
    return tags

def startThreads(fi_list, wpt, bcr, stok, thread_count=4, bow=False):
    sub_size = int(len(fi_list)/thread_count)
    
    for i in range(thread_count + 1):
        subs = fi_list[i*sub_size:(i+1)*sub_size]
        if i >= thread_count:
            subs = fi_list[i*sub_size:]
        
        thread = PipeThread(i, "Subset-{}".format(str(i)),
            subs, wpt, bcr, stok, bow)
        thread.start()

if __name__ == "__main__":
    
    main_folder = "/home/matthies/experiments/kodeAlltag/sample/"
    tok_model = "tokenizers/punkt/german.pickle"
    wdpath = "pickled_dicts/"
    regex = "[\w|�]+|[^\w\s]+"
    thread_count = 4
    bow = True

    
    m = getEmails(main_folder)
    
    wdo = WordDictObject.WordDictObject(wdpath)
    wpt = nltk.tokenize.RegexpTokenizer(regex)
    bcr = BadCharReplacer.BadCharReplacer(wdo)
    stok = nltk.data.load(tok_model)

    startThreads(m, wpt, bcr, stok, thread_count, bow)
