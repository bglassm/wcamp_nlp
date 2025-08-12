from pipeline.absa import classify_clauses
import pandas as pd
import config        # ← 여기 추가


df = pd.DataFrame({
    "review_id": [0, 1, 2],
    "clause": [
        "끝에 깨져왔네요 오이는 커요",    # 분명 부정
        "오이 그닥 신선하진 않아요..",     # 분명 부정
        "정말 만족합니다"         # 긍정
    ]
})
print(classify_clauses(df,
                       model_name=config.ABSA_MODEL_NAME,
                       batch_size=3,
                       device="cpu"))
