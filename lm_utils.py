from tqdm import tqdm
import torch
import openai
import os
import time
import numpy as np
import time
import wikipedia as wp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from llm import LLM

# import google.generativeai as genai #for gemini

device = "cuda"

def llm_init(model_name):
    global device
    global model
    global pipeline

    
    print("init model")
    if model_name == "aya_13b":
        device = "cuda"
        model = LLM(model="CohereForAI/aya-101", engine_dir="/data/aya-101-trt-bf16-engine/")
    
    if model_name == "chatgpt" or model_name == "gpt4":
        openai.api_key = os.getenv("OPENAI_API_KEY")

    # if model_name == "gemini":
    #     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    #     genai.configure(api_key=GOOGLE_API_KEY)

def wipe_model():
    global device
    global model
    global pipeline
    device = None
    model = None
    pipeline = None
    del device
    del model
    del pipeline

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 200):
    global model
    # if model_name == "gemini":
    #     model = genai.GenerativeModel("models/gemini-1.5-flash-001")
    #     response = model.generate_content(
    #         prompt,
    #         generation_config=genai.types.GenerationConfig(
    #             temperature=temperature,
    #             max_output_tokens=max_new_tokens
    #         )
    #     )
        
    #     if not response.candidates or not response.candidates[0].content.parts:
    #             print("Gemini blocked a response due to content moderation.")
    #             return "Error: Gemini blocked the response due to content moderation."

    #     generated_text = response.text.strip()
    #     token_probs = {}  # Gemini does not return token probabilities?

    #     if probs:
    #         return generated_text, token_probs
    #     else:
    #         return generated_text
        
    if model_name == "aya_13b":
        outputs = model.generate(prompt, max_new_tokens=max_new_tokens, return_dict=probs, temperature=temperature)

        # print(outputs)

        if isinstance(outputs, dict):
            return outputs["generated_texts"][0], outputs["token_probs"][0]
        else:
            return outputs[0]
    
    elif model_name == "chatgpt":
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            logprobs=1,
        )
        time.sleep(0.1)
        token_probs = {}
        for tok, score in zip(response.choices[0].logprobs.tokens, response.choices[0].logprobs.token_logprobs):
            token_probs[tok] = np.exp(score)
        if probs:
            return response.choices[0].text, token_probs
        else:
            return response.choices[0].text
    
    elif model_name == "gpt4":
        response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
                    logprobs=True,
                )
        time.sleep(0.1)
        token_probs = {}
        for thing in response['choices'][0]['logprobs']["content"]:
            token_probs[thing["token"]] = np.exp(thing["logprob"])
        if probs:
            return response['choices'][0]['message']['content'].strip(), token_probs
        else:
            return response['choices'][0]['message']['content'].strip()
    
def answer_parsing(response, model_name):
    # parsing special gemini answers
    # if model_name == "gemini":
    #     temp = response.lower().strip().split(" ")
    #     if temp[0] == "Error: ": # for content moderated response from Gemini
    #         return "Z" # so that its absolutely wrong
    #     for option in ["**a", "**b", "**c", "**d", "**e"]:
    #         for i in range(len(temp)):
    #             if option in temp[i]:
    #                 return option.replace("**", "").upper()
    # mode 1: answer directly after
    temp = response.strip().split(" ")
    for option in ["A", "B", "C", "D", "E"]:
        if option in temp[0]:
            return option
    # mode 2: "The answer is A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "the answer is " + option in temp:
            return option.upper()
    # mode 3: "Answer: A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "answer: " + option in temp:
            return option.upper()
    # mode 4: " A/B/C/D/E " or " A/B/C/D/E."
    for option in ["A", "B", "C", "D", "E"]:
        if " " + option + " " in response or " " + option + "." in response:
            return option
    # mode 5: "The correct answer is A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "the correct answer is " + option in temp:
            return option.upper()
    # mode 6: "A: " or "B: " or "C: " or "D: " or "E: "
    for option in ["A", "B", "C", "D", "E"]:
        if option + ": " in response:
            return option
    # mode 7: "A/B/C/D/E" and EOS
    try:
        for option in ["A", "B", "C", "D", "E"]:
            if option + "\n" in response or response[-1] == option:
                return option
    except:
        pass
    # mode 8: "true" and "false" instead of "A" and "B" for feedback abstention

    if "true" in response.lower():
        return "A"
    if "false" in response.lower():
        return "B"

    # fail to parse
    # print("fail to parse answer", response, "------------------")
    return "Z" # so that its absolutely wrong

prompt = "Question: Who is the 44th president of the United States?\nAnswer:"

# llm_init("aya_13b")
# answer = llm_response(prompt, "aya_13b", probs=False)
# print(answer)

text_classifier = None

def mlm_text_classifier(texts, labels, EPOCHS=10, BATCH_SIZE=32, LR=5e-5):
    # train a roberta-base model to classify texts
    # texts: a list of strings
    # labels: a list of labels of 0 or 1

    # load model
    global text_classifier
    text_classifier = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # tokenize
    encodeds = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodeds["input_ids"]
    attention_mask = encodeds["attention_mask"]

    # train
    optimizer = torch.optim.Adam(text_classifier.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    batch_size = BATCH_SIZE
    for epoch in tqdm(range(EPOCHS)):
        for i in range(0, len(input_ids), batch_size):
            optimizer.zero_grad()
            outputs = text_classifier(input_ids[i:i+batch_size], attention_mask=attention_mask[i:i+batch_size])
            logits = outputs.logits
            loss = loss_fn(logits, torch.tensor(labels[i:i+batch_size]))
            loss.backward()
            optimizer.step()

def text_classifier_inference(text):
    # provide predicted labels and probability
    # text: a string
    # return: label, probability
    global text_classifier

    assert text_classifier is not None, "text_classifier is not initialized"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text_classifier.eval()
    encodeds = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodeds["input_ids"]
    attention_mask = encodeds["attention_mask"]
    outputs = text_classifier(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return predictions[0].item(), probs[0][predictions[0]].item() # label, probability for the predicted label

# texts = ["I like this movie", "I hate this movie", "I like this movie", "I hate this movie"] * 100
# labels = [1, 0, 1, 0] * 100
# mlm_text_classifier(texts, labels)
# print(text_classifier_inference("I like this movie"))
# print(text_classifier_inference("I hate this movie"))

def get_wiki_summary(text):
    passage = ""
    try:
        for ent in wp.search(text[:100], results = 3):
            try:
                passage = "".join(wp.summary(ent, sentences=10)).replace("\n", " ")
            except:
                # print("error in retrieving summary for " + ent)
                pass
    except:
        print("error in wiki search")
        time.sleep(2)
        pass
    return passage