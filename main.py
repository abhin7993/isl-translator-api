from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from spacy_rules import translate_text, translate_to_tokens

app = FastAPI(
    title="ISL Translator API",
    description="Translates English text to Indian Sign Language (ISL) gloss using spaCy rules.",
    version="1.0.0",
)


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    input: str
    isl_gloss: str


class TokensResponse(BaseModel):
    input: str
    tokens: list[dict]


@app.get("/")
def root():
    return {"message": "ISL Translator API is running. Visit /docs for the API documentation."}


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest):
    """
    Translate English text to ISL gloss (space-separated words in ISL word order).
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    gloss = translate_text(request.text)
    return TranslateResponse(input=request.text, isl_gloss=gloss)


@app.post("/translate/tokens", response_model=TokensResponse)
def translate_tokens(request: TranslateRequest):
    """
    Translate English text and return detailed ISL token list with linguistic metadata.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    tokens = translate_to_tokens(request.text)
    token_list = [
        {
            "text": t.text,
            "orig_id": t.orig_id,
            "dep": t.dep,
            "head": t.head,
            "tag": t.tag,
            "ent_type": t.ent_type,
        }
        for t in tokens
    ]
    return TokensResponse(input=request.text, tokens=token_list)


@app.get("/translate", response_model=TranslateResponse)
def translate_get(text: str):
    """
    Translate English text to ISL gloss via GET request with a query parameter.
    Example: /translate?text=Hello+how+are+you
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    gloss = translate_text(text)
    return TranslateResponse(input=text, isl_gloss=gloss)
