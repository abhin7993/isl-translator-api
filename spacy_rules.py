# -*- coding: utf-8 -*-

import sys
import getopt
from spacy import load
from typing import NamedTuple
from nltk.corpus import wordnet


nlp = load("en_core_web_sm")

class ISLToken(NamedTuple):
    """Class to hold ISL token with relevant syntactic information"""

    text: str
    orig_id: int
    dep: str
    head: int
    tag: str
    ent_type: str
    children: list


def filter_spans(spans):
    """Filter a sequence of spans so they don't contain overlaps"""

    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def token_chunker(doc):
    """Merge entities and noun chunks into one token"""

    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)


def cc_chunker(doc):
    """
    Merge cc (only 'and' for now) conjunctions for like elements. To be run
    after token_chunker.
    returns -1 if and is chunked, or the token index if sentence is to be split
    """

    for token in doc:
        i = token.i
        if (token.text.lower() == "and") and (token.dep_ == "cc"):
            if i == 0:
                return 0

            if (token.head.i == i-1):
                and_span = doc[token.head.left_edge.i : token.head.right_edge.i + 1]
                with doc.retokenize() as retokenizer:
                    retokenizer.merge(and_span)
                return -1

            return i

        return -1


# list of articles etc. not present in ISL
droplist = {'be', 'do', 'a', 'the', 'of', 'for', 'from', 'to'}

# synonyms in ISL list for various common English words
worddict = {'there': ['her', 'she', 'that', 'it'], 'no': ['not', 'n\'t'],
 'possible': ['can', 'may']}

def find_syn(token):
    """Finds a synonym that exists in the available wordlist, from WordNet"""

    token_synsets = wordnet.synsets(token)
    for synset in token_synsets:
        for l in synset.lemma_names():
            if l in worddict:
                return l

    for key in worddict.keys():
        if token in worddict[key]:
            return key
    return token


def eng_isl_translate(doc):
    """Function to translate English to ISL gloss"""

    dep_list = []
    type_list = []
    tag_list = []
    ISLTokens = []
    done_list = []

    token_chunker(doc)

    doc2 = None
    and_tkn = None

    for token in doc:
        if "CC" == token.tag_ and "and" == token.text.lower():
            and_i = cc_chunker(doc)

            if and_i > -1:
                doc2_root_i = doc[and_i + 1 : ].root.i - and_i - 1
                doc2 = doc[and_i + 1 : ].as_doc()
                doc2[doc2_root_i].dep_ = "ROOT"
                and_tkn = doc[and_i]

                if and_i == 0:
                    ISLTokens2 = eng_isl_translate(doc2)
                    ISLTokens.append(and_tkn)
                    ISLTokens.extend(ISLTokens2)
                    return ISLTokens

                doc = doc[0 : and_i].as_doc()
                break

    for token in doc:
        dep_list.append(token.dep_)
        tag_list.append(token.tag_)
        type_list.append(token.ent_type_)

    # DATE goes first in ISL sentences
    if "DATE" in type_list:
        date_i = type_list.index("DATE")
        done_list.append(date_i)
        tkn = doc[date_i]
        ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                tkn.ent_type_, [child for child in tkn.children]))
        if doc[date_i].dep_ == "pobj":
            date_ii = doc[date_i].head.i
            tkn = doc[date_ii]
            done_list.append(date_ii)
            ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i,
                    tkn.tag_, tkn.ent_type_,
                    [child for child in tkn.children]))

    # Subject comes next
    if "nsubj" in dep_list:
        nsubj_i = dep_list.index("nsubj")
        tkn = doc[nsubj_i]
        if not tkn.tag_[0] == 'W' and tkn.i not in done_list:
            done_list.append(nsubj_i)
            ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                    tkn.ent_type_, [child for child in tkn.children]))

    if "ROOT" not in dep_list:
        return ISLTokens

    root_i = dep_list.index("ROOT")
    root_children = [child for child in doc[root_i].children]

    # Complements and prepositional phrases
    if not {"xcomp", "ccomp", "prep", "advcl"}.isdisjoint([child.dep_ for child in doc[root_i].children]):
        for child in doc[root_i].children:
            if child.dep_ in ("xcomp", "ccomp", "prep", "advcl"):
                subtree_span = doc[child.left_edge.i : child.right_edge.i + 1]
                for tkn in subtree_span:
                    if tkn.i not in done_list:
                        ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_,
                                tkn.head.i, tkn.tag_, tkn.ent_type_,
                                [child for child in tkn.children]))
                        done_list.append(tkn.i)

    # Direct object comes before root verb
    if "dobj" in [child.dep_ for child in doc[root_i].children]:
        dobj_i_1 = [child.dep_ for child in doc[root_i].children].index("dobj")
        dobj_i = root_children[dobj_i_1].i
        tkn = doc[dobj_i]
        if dobj_i not in done_list:
            done_list.append(dobj_i)
            ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                    tkn.ent_type_, [child for child in tkn.children]))

    # Root verb
    tkn = doc[root_i]
    done_list.append(root_i)
    ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
            tkn.ent_type_, [child for child in tkn.children]))
    isl_root_i = len(ISLTokens) - 1

    # Auxiliaries after the verb
    if "aux" in [child.dep_ for child in doc[root_i].children]:
        aux_i_1 = [child.dep_ for child in doc[root_i].children].index("aux")
        aux_i = root_children[aux_i_1].i
        tkn = doc[aux_i]
        done_list.append(aux_i)
        ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                tkn.ent_type_, [child for child in tkn.children]))

    # Negatives come last
    if "neg" in dep_list:
        neg_i = dep_list.index("neg")
        tkn = doc[neg_i]
        done_list.append(neg_i)
        ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                tkn.ent_type_, [child for child in tkn.children]))

    # Question words come last
    if tag_list and tag_list[0][0] == 'W':
        tkn = doc[0]
        done_list.append(0)
        ISLTokens.append(ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                tkn.ent_type_, [child for child in tkn.children]))

    j = isl_root_i
    for tkn in root_children:
        if tkn.i not in done_list and tkn.dep_ not in ["aux", "punct", "neg"]:
            done_list.append(tkn.i)
            ISLTokens.insert(j, ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i,
                    tkn.tag_, tkn.ent_type_,
                    [child for child in tkn.children]))
            j += 1
    for tkn in doc:
        if tkn.i not in done_list and tkn.dep_ not in ["aux", "punct", "neg"]:
            done_list.append(tkn.i)
            ISLTokens.insert(j, ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i,
                    tkn.tag_, tkn.ent_type_,
                    [child for child in tkn.children]))
            j += 1

    # Drop tokens not used in ISL
    ISLTokens[:] = [isl_tkn for isl_tkn in ISLTokens if isl_tkn.text not in droplist]

    if doc2:
        ISLTokens2 = eng_isl_translate(doc2)
        ISLTokens.append(and_tkn)
        ISLTokens.extend(ISLTokens2)

    return ISLTokens


def translate_to_tokens(text):
    """Convert English text to ISLToken list"""

    doc = nlp(text)
    ISLTknOP = []

    for sent in doc.sents:
        ISLSent = eng_isl_translate(sent.as_doc())
        ISLTknOP.extend(ISLSent)

    return ISLTknOP


def translate_text(text):
    """Convert English text to space-separated ISL gloss"""

    raw_token_list = translate_to_tokens(text)
    raw_isl_text = " ".join([isl_tkn.text.lower() for isl_tkn in raw_token_list])
    return raw_isl_text


def main(argv):
    text = ''

    try:
        opts, args = getopt.getopt(argv, "ht:", ["help", "text="])
    except getopt.GetoptError:
        print('spacy_rules.py -t <English text>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print('spacy_rules.py -t <English text>')
            sys.exit()
        elif opt in ("-t", "--text"):
            text = arg

    if not text:
        text = "Where is Sanket going?"

    print(translate_text(text))


if __name__ == "__main__":
    main(sys.argv[1:])
