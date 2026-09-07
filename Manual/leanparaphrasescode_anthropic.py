from anthropic import Anthropic

client = Anthropic()

dataset = open("leanparaphrasesdataset.txt", "r")

for i in range(100):
    problem = dataset.readline()
    message = "Formalize this natural language problem into Lean 4 language. Only create the prompt, and use 'sorry' as the proof. Your response should only include the final formalized Lean 4 statement.\n" + problem

    result = client.messages.create(
        model="claude-opus-5",
        max_tokens=10000,
        thinking={
            "type": "disabled"
        },
        messages=[
            {"role": "user", "content": message}
        ]
        )
    modelanswer = result.content[0].text

    print(message)
    print()
    print(modelanswer)
    print("--------------------")