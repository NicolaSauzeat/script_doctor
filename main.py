# This is a sample Python script.

from gensim import corpora, models
import pandas as pd
# list_of_list_of_tokens = [["a","b","c"], ["d","e","f"]]
# ["a","b","c"] are the tokens of document 1, ["d","e","f"] are the tokens of document 2...
import re
from collections import defaultdict, Counter


def extract_scene(movie_title, script):
    scenes = re.split(r"INT|EXT", script)
    return scenes

def extract_dialog_2(movie_title, script):
    print("coucou")
    for match_base in re.findall(r"^[ \t]*[A-Z]{3,}\W*?(?:[\r\n]+(.+?)[\r\n][\s\t]*[\r\n]|\:(.+?)[\r\n])",
                                                script, flags=re.DOTALL | re.MULTILINE):
        for i in range(0, len(match_base)):
            if len(match_base[i]) > 0:
                dialogue = ""
                text = match_base[i].strip()
                dialogue += text
        #character = character.strip()



def extract_dialog(movie_title, script):
    dialog_for_movie = defaultdict(lambda: [])
    global counter
    # consider dialog to ONLY be first paragraph of text after a character name introduced or
    # inline if colon after character name
    for match_base, character in zip(re.findall(r"^[ \t]*[A-Z]{3,}\W*?(?:[\r\n]+(.+?)[\r\n][\s\t]*[\r\n]|\:(.+?)[\r\n])", script,
                                 flags=re.DOTALL | re.MULTILINE), re.findall(r"^[ \t]*[A-Z:]{3,}", script,
                                 flags=re.DOTALL)):
            dialogue = ""
            for i in range(0, len(match_base)):
                if len(match_base[i]) >0:
                    text = match_base[i].strip()
                    dialogue += text
            character = character.strip()
            dialog_for_movie[movie_title].append({dialogue, character})

    if len(dialog_for_movie[0]) <1 :
        test=  extract_dialog_2(movie_title, script)
    return dialog_for_movie

    def title_cleaner(x):
        return re.sub(r"[^\w ]", "", x.lower())




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test = pd.read_csv("/home/nicolas/Bureau/pythonProject/Script Doctor/Resultats/in/train_df.csv", sep=",")
    found_scripts = 0
    essai = defaultdict(lambda: [])
    for script_name, i in zip(test["Script"], test.index):
        movie_name = script_name.split("\n")[0].replace(" ", "").replace("\t", "")
        found_scripts += 1
        # process the script
        test = extract_dialog(movie_name, script_name)

    print("hi")


"""
        match = match_base[0]
        if len(match_base[0]) == 0:
            match = match_base[1]
        if re.search(r"\b(you|i|we|us|he|her)\b", match.lower()):
            base = re.sub(r"(\w+)['\"“”‘’„”](\w)", "\\1\\2", match.lower())
            # splinter a long run of text into sentences.
            for p1 in re.split(r"[\.\?\!]+", base):
                #             for p1 in nltk.sent_tokenize(base):
                simple_text = re.sub(r" {2,}", " ", re.sub(r"[^\w ]", " ", p1))
                if len(simple_text) > 0:
                    dialog_for_movie[movie_title].append(simple_text.strip().split(" "))
"""
