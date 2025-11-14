import pandas as pd

df=pd.read_csv("all_model_moderation_results.csv")

asr_summary=df.groupby("model")["llamaguard3_label"].agg(
    n_attacks="count",
    n_jailbreaks="sum"
).reset_index()

asr_summary["ASR"]=asr_summary["n_jailbreaks"]/asr_summary["n_attacks"]
asr_summary["ASR"] = asr_summary["ASR"].round(3)
asr_summary.to_csv("asr_summary.csv",index=False)