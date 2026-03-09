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
    orig_text: str = ''


def make_isl_token(tkn):
    """Create ISLToken from a spaCy Token, preserving original text."""
    return ISLToken(tkn.lemma_, tkn.i, tkn.dep_, tkn.head.i, tkn.tag_,
                    tkn.ent_type_, [child for child in tkn.children], tkn.text)


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
                    ISLTokens.append(make_isl_token(and_tkn))
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
        ISLTokens.append(make_isl_token(tkn))
        if doc[date_i].dep_ == "pobj":
            date_ii = doc[date_i].head.i
            tkn = doc[date_ii]
            done_list.append(date_ii)
            ISLTokens.append(make_isl_token(tkn))

    # Subject comes next
    if "nsubj" in dep_list:
        nsubj_i = dep_list.index("nsubj")
        tkn = doc[nsubj_i]
        if not tkn.tag_[0] == 'W' and tkn.i not in done_list:
            done_list.append(nsubj_i)
            ISLTokens.append(make_isl_token(tkn))

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
                        ISLTokens.append(make_isl_token(tkn))
                        done_list.append(tkn.i)

    # Direct object comes before root verb
    if "dobj" in [child.dep_ for child in doc[root_i].children]:
        dobj_i_1 = [child.dep_ for child in doc[root_i].children].index("dobj")
        dobj_i = root_children[dobj_i_1].i
        tkn = doc[dobj_i]
        if dobj_i not in done_list:
            done_list.append(dobj_i)
            ISLTokens.append(make_isl_token(tkn))

    # Root verb
    tkn = doc[root_i]
    done_list.append(root_i)
    ISLTokens.append(make_isl_token(tkn))
    isl_root_i = len(ISLTokens) - 1

    # Auxiliaries after the verb
    if "aux" in [child.dep_ for child in doc[root_i].children]:
        aux_i_1 = [child.dep_ for child in doc[root_i].children].index("aux")
        aux_i = root_children[aux_i_1].i
        tkn = doc[aux_i]
        done_list.append(aux_i)
        ISLTokens.append(make_isl_token(tkn))

    # Negatives come last
    if "neg" in dep_list:
        neg_i = dep_list.index("neg")
        tkn = doc[neg_i]
        done_list.append(neg_i)
        ISLTokens.append(make_isl_token(tkn))

    # Question words come last
    if tag_list and tag_list[0][0] == 'W':
        tkn = doc[0]
        done_list.append(0)
        ISLTokens.append(make_isl_token(tkn))

    j = isl_root_i
    for tkn in root_children:
        if tkn.i not in done_list and tkn.dep_ not in ["aux", "punct", "neg"]:
            done_list.append(tkn.i)
            ISLTokens.insert(j, make_isl_token(tkn))
            j += 1
    for tkn in doc:
        if tkn.i not in done_list and tkn.dep_ not in ["aux", "punct", "neg"]:
            done_list.append(tkn.i)
            ISLTokens.insert(j, make_isl_token(tkn))
            j += 1

    # Drop tokens not used in ISL
    ISLTokens[:] = [isl_tkn for isl_tkn in ISLTokens if isl_tkn.text not in droplist]

    if doc2:
        ISLTokens2 = eng_isl_translate(doc2)
        ISLTokens.append(make_isl_token(and_tkn))
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


def get_role(token):
    """Map ISLToken dep/tag/ent_type to a human-readable grammatical role."""
    # Named entities recognized by spaCy
    if token.ent_type in ('PERSON', 'ORG', 'GPE', 'LOC'):
        return 'name'
    # Proper noun POS tags (catches names spaCy NER misses)
    if token.tag in ('NNP', 'NNPS'):
        return 'name'
    # Heuristic: capitalized original text is likely a name (catches names spaCy misclassifies)
    # Skip tags that are clearly not names, and handle sentence-initial capitalization
    orig = token.orig_text or token.text
    _non_name_tags = ('PRP', 'PRP$', 'MD', 'DT', 'IN', 'CC', 'TO', 'EX')
    if orig and token.tag not in _non_name_tags and not token.tag.startswith(('VB', 'W', 'RB')):
        words = orig.split()
        if len(words) == 1:
            # Single word: only trust capitalization if NOT at sentence start (orig_id > 0)
            if token.orig_id > 0 and orig[0].isupper():
                return 'name'
        else:
            # Multi-word: check if any word after the first is capitalized (ignoring articles)
            if any(w[0].isupper() for w in words[1:] if w.lower() not in ('a', 'an', 'the')):
                return 'name'
    if token.dep in ('nsubj', 'nsubjpass'):
        return 'subject'
    if token.dep == 'dobj':
        return 'object'
    if token.dep == 'neg':
        return 'negation'
    if token.dep == 'aux':
        return 'auxiliary'
    if token.tag.startswith('W'):
        return 'question_word'
    if token.dep == 'ROOT' and token.tag.startswith('VB'):
        return 'verb'
    if token.tag.startswith('VB'):
        return 'verb'
    if token.tag.startswith('JJ'):
        return 'adjective'
    if token.tag.startswith('RB'):
        return 'adverb'
    if token.tag in ('NN', 'NNS'):
        return 'noun'
    if token.tag == 'PRP' or token.tag == 'PRP$':
        return 'pronoun'
    if token.tag == 'CC':
        return 'conjunction'
    if token.tag == 'CD':
        return 'number'
    if token.tag == 'IN':
        return 'preposition'
    return 'other'


def get_pos(tag):
    """Map spaCy tag to a readable part-of-speech label."""
    if tag.startswith('VB'):
        return 'verb'
    if tag.startswith('NN'):
        return 'noun'
    if tag.startswith('JJ'):
        return 'adjective'
    if tag.startswith('RB'):
        return 'adverb'
    if tag.startswith('PRP'):
        return 'pronoun'
    if tag.startswith('W'):
        return 'question_word'
    if tag == 'CC':
        return 'conjunction'
    if tag == 'CD':
        return 'number'
    if tag == 'IN':
        return 'preposition'
    if tag == 'MD':
        return 'modal'
    return 'other'


def translate_text_detailed(text):
    """Convert English text to ISL gloss with token role details."""
    raw_token_list = translate_to_tokens(text)

    gloss = " ".join([isl_tkn.text.lower() for isl_tkn in raw_token_list])

    tokens = []
    for tkn in raw_token_list:
        tokens.append({
            "word": tkn.text.lower(),
            "role": get_role(tkn),
            "pos": get_pos(tkn.tag)
        })

    return gloss, tokens


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

    gloss, tokens = translate_text_detailed(text)
    print(f"Gloss: {gloss}")
    for t in tokens:
        print(f"  {t['word']:20s} role={t['role']:15s} pos={t['pos']}")


if __name__ == "__main__":
    main(sys.argv[1:])
