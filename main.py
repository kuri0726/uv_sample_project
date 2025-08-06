# import dspy
# from dspy.evaluate import Evaluate
# from dspy.teleprompt import BootstrapFewShot

# # モデルを選択（OpenAIなど）
# dspy.settings.configure(openai_model="gpt-4")

# # モジュール（例えば質問応答）を定義


# class QAModule(dspy.Signature):
#     context = dspy.InputField()
#     question = dspy.InputField()
#     answer = dspy.OutputField()


# # 最適化対象のプロンプト定義
# qa_module = dspy.Predict(QAModule)

# # チューニング用データ
# train_data = [
#     {
#         "context": "富士山は日本で最も高い山です。静岡県と山梨県の境にあります。",
#         "question": "富士山はどこにありますか？",
#         "answer": "静岡県と山梨県の境にあります。",
#     },
#     {
#         "context": "東京は日本の首都であり、人口が最も多い都市です。",
#         "question": "日本の首都はどこですか？",
#         "answer": "東京です。",
#     },
# ]

# # 最適化エンジンでFew-shotプロンプトをブートストラップ（自動設計）
# teleprompter = BootstrapFewShot(metric="exact_match")
# optimized_module = teleprompter.compile(qa_module, train_data)

# # 最適化されたプロンプトで推論
# result = optimized_module(context="富士山は...", question="富士山の位置は？")
# print(result.answer)

import os

import dspy
from dotenv import load_dotenv

load_dotenv()


lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("API_KEY"))
dspy.configure(lm=lm)

math = dspy.ChainOfThought("question -> answer")
res = math(question="Two dice are tossed. What is the probability that the sum equals two?")

print(res)