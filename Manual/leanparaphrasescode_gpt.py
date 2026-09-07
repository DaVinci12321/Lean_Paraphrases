from openai import OpenAI

client = OpenAI()

dataset = open("leanparaphrasesdataset.txt", "r")

for i in range(100):
    problem = dataset.readline()
    message = "Formalize this natural language problem into Lean 4 language. Only create the prompt, and use 'sorry' as the proof. Your response should only include the final formalized Lean 4 statement.\n" + problem
    result = client.chat.completions.create(
        model="gpt-5.6-sol",
        messages=[
            {"role": "user", "content": message}
        ]
    )

    modelanswer=result.choices[0].message.content
    print(message)
    print()
    print(modelanswer)
    print("--------------------")