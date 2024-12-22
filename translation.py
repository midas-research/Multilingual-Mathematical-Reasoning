# Code to translate the English dataset to Hindi
# Used for Bilingual Dataset Creation

# I will be using multiple clients that have multiple api keys. The next client will take over when the previous one is exhausted


from together import Together
import json
import os


f = open('<dataset path>')

client1 = Together(api_key='api_key_1')
client2 = Together(api_key='api_key_2')
client3 = Together(api_key='api_key_3')
client4 = Together(api_key='api_key_4')
client5 = Together(api_key='api_key_5')
client6 = Together(api_key='api_key_6')


model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

def run_translation_task(client, question, answer):

  user_prompt = f""" Translate the following English Mathematics Question and its English Answer into Hindi. \n
  Note: Do not translate the Asymptote Language used within the text. \n
  ### English Mathematics Question: {question}
  ### English Mathematics Answer: {answer}
  \n Give the response in the following format: \n
  ### Translated Hindi Mathematics Question: [your response]
  ### Translated Hindi Mathematics Answer: [your response]
  """

  response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": user_prompt}],
  )
  return response


def translate_to_hindi_llama405B(client, question, answer):
  ind1=-1
  ind2=-1
  search_string1=''
  search_string2=''
  while ind1==-1 or ind2==-1:

    response = run_translation_task(client, question, answer)
    res = response.choices[0].message.content

    search_string1='Hindi Mathematics Question:'
    ind1 = res.find(search_string1)

    search_string2='### Translated Hindi Mathematics Answer:'
    ind2 = res.find(search_string2)


  return res[ind1+len(search_string1):ind2], res[ind2+len(search_string2):]


clients=[client4, client3, client2, client5, client1, client6]
data=json.load(f)
i=0
client_index=0
translated_hindi=[]
for sample in data:

  question = sample['question']
  answer = sample['answer']
  try:
    translated_question, translated_answer = translate_to_hindi_llama405B(clients[client_index], question, answer)
    i+=1
  except Exception as e:
    print(e)
    print(i)
    if client_index==len(clients)-1:
      client_index=0
    else:
      client_index+=1

    translated_question, translated_answer = translate_to_hindi_llama405B(clients[client_index], question, answer)
    i+=1

  translated_sample=sample.copy()
  translated_sample['question']=translated_question
  translated_sample['answer']=translated_answer
  translated_hindi.append(translated_sample)


print(i)
print(translated_hindi)
f.close()

with open("hindi_translated.json", "w", encoding='utf8') as outfile:
    json.dump(translated_hindi, outfile, ensure_ascii=False)