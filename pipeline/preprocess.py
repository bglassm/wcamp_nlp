import logging, re, pandas as pd
import config

logger = logging.getLogger(__name__)

_RE_LAUGH     = re.compile(r"(ㅋ)\1{2,}")  
_RE_EXCLAIM   = re.compile(r"(!|\?){2,}")  
_RE_EMOJI_ETC = re.compile(r"[^\w\s\uAC00-\uD7A3.!?]+")

def _clean_text(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    txt = txt.strip()
    txt = _RE_LAUGH.sub(r"\1\1", txt)
    txt = _RE_EXCLAIM.sub(r"\1", txt)
    txt = _RE_EMOJI_ETC.sub("", txt)
    return txt

def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    if "review" not in df.columns:
        raise KeyError("Run loader.load_reviews first")

    logger.info("🔧 Pre-processing %s rows", f"{len(df):,}")
    out = df.copy()
    out["review"] = out["review"].astype(str).map(_clean_text)

    before = len(out)
    out = out.drop_duplicates(subset="review")
    out = out.loc[out["review"].str.len() > 0].reset_index(drop=True)
    logger.info("✨ Completed: %s → %s rows after cleaning",
                f"{before:,}", f"{len(out):,}")
    return out
