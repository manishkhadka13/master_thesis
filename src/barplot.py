import pandas as pd
import matplotlib.pyplot as plt

asr_summary = pd.read_csv("asr_summary.csv")

plt.figure(figsize=(10,6))
bars = plt.bar(asr_summary['model'], asr_summary['ASR']*100, color='royalblue')
plt.ylabel("Attack Success Rate (%)")
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.tight_layout()


for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.1f}%', 
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=10)

plt.savefig("asr_per_model.png", dpi=300)
plt.close()
