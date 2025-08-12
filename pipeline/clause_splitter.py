# pipeline/clause_splitter.py

from typing import List
import pandas as pd
import kss

def split_clauses(
    df: pd.DataFrame,
    text_col: str,
    connectives: List[str],
    id_col: str = "review_id",
) -> pd.DataFrame:
    """
    원본 리뷰 DataFrame → 절 단위 DataFrame
    반환 컬럼: [id_col, "clause"]
    """
    rows: list[dict] = []
    for _, row in df.iterrows():
        rid = row[id_col]
        sent = row[text_col]
        # 1) KSS로 문장 분리
        sents = kss.split_sentences(sent)
        # 2) 역접 접속사 기준으로 further split
        clauses: list[str] = []
        for s in sents:
            parts = [s]
            for conn in connectives:
                new_parts: list[str] = []
                for p in parts:
                    new_parts.extend(p.split(conn))
                parts = new_parts
            clauses.extend([c.strip() for c in parts if c.strip()])
        # 3) DataFrame 행으로 추가
        for clause in clauses:
            rows.append({id_col: rid, "clause": clause})
    return pd.DataFrame(rows)
