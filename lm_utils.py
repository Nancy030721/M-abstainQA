from tqdm import tqdm
import torch
import time
import numpy as np
import time
import os
import wikipedia as wp
from transformers import AutoTokenizer, AutoModelForSequenceClassification


device = "cuda"

def llm_init(model_name):
    global device
    global model
    global pipeline
    global openai_client

    
    print("init model")
    if model_name == "aya_13b":
        device = "cuda"
        from enc_dec_model import EncDecModel
        model = EncDecModel(model="/data/aya-101", engine_dir="/data/aya-101-trt-bf16-engine-m/")
    
    if model_name == "gpt4":
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if model_name == "gemma":
        device = "cuda"
        os.environ["VLLM_ATTENTION_BACKEND"] = "flashinfer"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
        from vllm import LLM
        model = LLM(model="/data/gemma-3-27b-it",  tensor_parallel_size=4)


def wipe_model():
    global device
    global model
    global pipeline
    global openai_client
    device = None
    model = None
    pipeline = None
    openai_client = None
    del device
    del model
    del pipeline
    del openai_client

def llm_response(prompt, model_name, probs = False, temperature = 1.0, repetition_penalty=1.0, max_new_tokens = 200):
    global model

    if model_name == "aya_13b":
        return model.generate(prompt, max_new_tokens=max_new_tokens, return_dict=probs, temperature=temperature, repetition_penalty=repetition_penalty)

    elif model_name == "gemma":
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            logprobs=1 if probs else None
        )

        responses = model.generate(prompt, sampling_params)

        generated_texts = []
        token_probs_list = []
        for response in responses:
            gen_text = response.outputs[0].text.strip()
            generated_texts.append(gen_text)

            token_probs = {}
            if probs and response.outputs[0].logprobs is not None:
                for logprob_dict in response.outputs[0].logprobs:
                    for _, logprob_obj in logprob_dict.items():
                        token = logprob_obj.decoded_token
                        token_probs[token] = np.exp(logprob_obj.logprob)
            token_probs_list.append(token_probs)

        if probs:
            return {"generated_texts": generated_texts, "token_probs": token_probs_list}
        else:
            return generated_texts

    elif model_name == "gpt4":
        generated_texts = []
        token_probs_list = []
        for single_prompt in prompt:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                temperature=temperature,
                messages=[
                    {"role": "user", "content": single_prompt}
                ],
                max_tokens=max_new_tokens,
                logprobs=True,
            )
            time.sleep(0.1)
            token_probs = {}
            for token_log in response.choices[0].logprobs.content:
                token_probs[token_log.token] = np.exp(token_log.logprob)
            generated_text = response.choices[0].message.content.strip()
            generated_texts.append(generated_text)
            token_probs_list.append(token_probs)

        if probs:
            return {"generated_texts": generated_texts, "token_probs": token_probs_list}
        else:
            return generated_texts
    
def answer_parsing(response, model_name):
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