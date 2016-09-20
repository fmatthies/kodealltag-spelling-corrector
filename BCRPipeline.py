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
import argparse


class PipelineParser(argparse.ArgumentParser):
    def __init__(self):
        argparse.ArgumentParser.__init__(self, description="Usenet E-Mail Tokenizer/Lemmatizer")
        self.add_argument(
            'input-folder',
            type = str, nargs = 1, help = 'path to parent email folder'
        )
        self.add_argument(
            '-o', '--out-folder', metavar = 'Output-Folder',
            action = 'store', nargs = '?', default = None,
            type = str, help = 'If this argument is given, the produced ".tok" files are stored here; either relative or absolute (default: they are stored alongside their respective original txt files)'
        )
        self.add_argument(
            '-s', '--sent-tok', metavar = 'Sentence Tokenizer',
            action = 'store', nargs = '?', default = 'german',
            type = str, help = 'A Tokenizer model (anything that is accesible from NLTK is possible (default: german)'
        )
        self.add_argument(
            '-d', '--word-dict', metavar = 'Word Dictionary',
            action = 'store', nargs = '?', default = 'pickled_dicts/german/',
            type = str, help = 'A folder with pickled dictionary items; if this is given, bad characters (e.g. �) are replaced with correct characters according to the BadCharReplacer algorithm that utilizes word unigrams and Edit Distance (default: pickled_dicts/german/)'
        )
        self.add_argument(
            '-r', '--tok-regex', metavar = 'Preliminary Wordtokenizer',
            action = 'store', nargs = '?', default = '[\w|�]+|[^\w\s]+',
            type = str, help = 'A regular expression that is used for preliminary word tokenization; these tokens are given to the BadCharReplacer (default: "[\w|�]+|[^\w\s]+" )'
        )
        self.add_argument(
            '-t', '--threads', metavar = 'Number of Threads',
            action = 'store', nargs = '?', default = 1,
            type = int, help = 'Number of threads to start (default: 1)'
        )
        self.add_argument(
            '-b', '--bag-of-words', metavar = 'Bag-of-Words file',
            action = 'store', nargs = '?', default = False,
            type = bool, help = 'If true, there will also be ".lem" files created that only contains whitespace delimited lemma of an email (default: False)'
        )
        

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
    def __init__(self, threadID, name, subset, pretokenizer, charrepl, stok, ifoo, bow=False, out_folder=None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.subset = subset
        self.pretok = pretokenizer
        self.charrepl = charrepl
        self.sent_tokenizer = stok
        self.writeBoW = bow
        self.outFolder = out_folder
        self.main_folder = os.path.abspath(ifoo)

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
        out = mail
        if self.out_folder:
            out = os.path.relpath(mail, self.main_folder)
            os.path.join(self.out_folder, out)
            os.makedirs(os.path.dirname(out), exist_ok=True)

        with open(out+".tok", "w") as tfile:
            tfile.write("\n\n".join(tagged))
        if self.writeBoW:
            with open(out+".lem", "w") as lfile:
                for sent in tagged:
                    lines = sent.split("\n")
                    for line in lines:
                        line = line.split("\t")
                        if len(line) > 1:
                            if not re.search("(\$.|\$\()", line[1]):
                                lfile.write("{} ".format(line[-1]).lower())

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

def startThreads(fi_list, wpt, bcr, stok, ifoo, thread_count=4, bow=False, out_folder=None):
    sub_size = int(len(fi_list)/thread_count)
    
    for i in range(thread_count + 1):
        subs = fi_list[i*sub_size:(i+1)*sub_size]
        if i >= thread_count:
            subs = fi_list[i*sub_size:]
        
        thread = PipeThread(i, "Subset-{}".format(str(i)),
            subs, wpt, bcr, stok, ifoo, bow, out_folder)
        thread.start()

if __name__ == "__main__":
    parser = PipelineParser()
    args = vars(parser.parse_args())

    mails = getEmails(args['input-folder'][0])
    wrd_dict = WordDictObject.WordDictObject(args['word_dict'])
    prel_tok = nltk.tokenize.RegexpTokenizer(args['tok_regex'])
    sent_tok = nltk.data.load(
        os.path.join('tokenizers','punkt','{}.pickle'.format(args['sent_tok']))
    )
    chr_repl = BadCharReplacer.BadCharReplacer(wrd_dict)
    
    
    startThreads(mails, prel_tok, chr_repl, sent_tok,
        in_folder = args['input-folder'][0],
        thread_count = args['threads'],
        bow = args['bag_of_words'],
        out_folder = args['out_folder']
    )
