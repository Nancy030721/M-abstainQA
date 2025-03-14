#!/usr/bin/env python3
import json
import argparse
import random
import time
import lm_utils
import metrics
from tqdm import tqdm
import os

BATCH_SIZE = 4

def run_pipeline(model_name, dataset_name, source_language, portion, local_out, feedback_out, result_out, batch_size):
    # 3 related languages for each target language
    language_related = {
        "en": ["German", "Dutch", "French"],
        "ru": ["Ukrainian", "Romanian", "Catalan"],
        "de": ["Dutch", "English", "French"],
        "zh": ["Arabic", "Slovak", "Danish"],
        "fr": ["Catalan", "German", "Spanish"],
        "es": ["Catalan", "Romanian", "French"],
        "it": ["Catalan", "Romanian", "Ukrainian"],
        "nl": ["German", "Italian", "Ukrainian"],
        "vi": ["Indonesian", "English", "Bengali"],
        "id": ["Vietnamese", "Catalan", "Russian"],
        "ar": ["Chinese", "Slovak", "Danish"],
        "hu": ["Romanian", "German", "French"],
        "ro": ["Catalan", "Italian", "Spanish"],
        "da": ["Slovak", "Dutch", "Ukrainian"],
        "sk": ["Chinese", "Arabic", "Danish"],
        "uk": ["Russian", "Italian", "Croatian"],
        "ca": ["Romanian", "Spanish", "Italian"],
        "sr": ["Slovak", "Danish", "Croatian"],
        "hr": ["Ukrainian", "Italian", "Dutch"],
        "hi": ["Bengali", "Talugu", "Marathi"],
        "bn": ["Hindi", "Telugu", "Nepali"],
        "ta": ["Malayalam", "Marathi", "Kannada"],
        "ne": ["Kanaada", "Telugu", "Hindi"],
        "ml": ["Tamil", "Marathi", "Kannada"],
        "mr": ["Tamil", "Malayalam", "Hindi"],
        "te": ["Kannada", "Tamil", "Nepali"],
        "kn": ["Telugu", "Malaayalam", "Tamil"]
    }

    filepath = f"data/{dataset_name}/{dataset_name}_{source_language}.json"
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Keep portion of dev and test data
    portion_dev_count = int(len(data["dev"]) * portion)
    portion_test_count = int(len(data["test"]) * portion)
    data["dev"] = data["dev"][:portion_dev_count]
    data["test"] = data["test"][:portion_test_count]

    correct_flags = []
    answers_given = []

    print("1: Generating answers for each test question.")
    test_prompts = []
    for instance in data["test"]:
        prompt_text = f"Question: {instance['question']}\n"
        for choice_key, choice_text in instance["choices"].items():
            prompt_text += f"{choice_key}: {choice_text}\n"
        prompt_text += "Choose one answer from the above choices. Just provide the letter (A, B, C, or D) of the correct choice. The answer is"
        test_prompts.append(prompt_text)

    for i in tqdm(range(0, len(test_prompts), batch_size)):
        batch_prompts = test_prompts[i:i+batch_size]
        # batch_answers: List[str] (batch_size, )
        batch_answers = lm_utils.llm_response(batch_prompts, model_name, probs=False, max_new_tokens=10)
        for answer_text, instance in zip(batch_answers, data["test"][i:i+batch_size]):
            label = lm_utils.answer_parsing(answer_text, model_name)
            correct_flags.append(1 if label == instance["answer"] else 0)
            answers_given.append(answer_text)

    print("\n2: Generating multilingual feedback for each answer.")
    feedback_prompts = []
    feedback_instance_indices = []
    feedback_language_indices = []
    for i, instance in enumerate(data["test"]):
        base_prompt = f"Question: {instance['question']}\n"
        for ck, ct in instance["choices"].items():
            base_prompt += f"{ck}: {ct}\n"
        base_prompt += (f"Choose one answer from the above choices. Proposed answer: {answers_given[i].strip()}\n"
                        "Please review the proposed answer and provide a paragraph of feedback on its correctness. "
                        "Feedback should be in <LANG>.\nFeedback:")
        related_list = language_related.get(source_language, ["English", "English", "English"])
        for lang_idx in range(3):
            prompt_lang = base_prompt.replace("<LANG>", related_list[lang_idx])
            feedback_prompts.append(prompt_lang)
            feedback_instance_indices.append(i)
            feedback_language_indices.append(lang_idx)

    feedback_responses = [None] * len(feedback_prompts)
    for i in tqdm(range(0, len(feedback_prompts), batch_size)):
        batch_prompts = feedback_prompts[i:i+batch_size]
        # batch_feedback: List[str] (batch_size, )
        batch_feedback = lm_utils.llm_response(batch_prompts, model_name, probs=False, temperature=0.7, repetition_penalty=1.1)
        feedback_responses[i:i+batch_size] = batch_feedback

    feedback1 = [None] * len(data["test"])
    feedback2 = [None] * len(data["test"])
    feedback3 = [None] * len(data["test"])
    for idx, resp in enumerate(feedback_responses):
        inst_idx = feedback_instance_indices[idx]
        lang_idx = feedback_language_indices[idx]
        cleaned_resp = resp.split("\n")[0].strip() if resp.strip() else "No feedback provided."
        if lang_idx == 0:
            feedback1[inst_idx] = cleaned_resp
        elif lang_idx == 1:
            feedback2[inst_idx] = cleaned_resp
        elif lang_idx == 2:
            feedback3[inst_idx] = cleaned_resp

    print("\n3: Make abstain decision based on feedback.")
    final_prompts = []
    for i, instance in enumerate(data["test"]):
        combined_prompt = f"Question: {instance['question']}\n"
        for ck, ct in instance["choices"].items():
            combined_prompt += f"{ck}: {ct}\n"
        combined_prompt += (f"Choose one answer. Proposed answer: {answers_given[i].strip()}\n\n"
                             f"Feedback 1: {feedback1[i].strip()}\n\n"
                             f"Feedback 2: {feedback2[i].strip()}\n\n"
                             f"Feedback 3: {feedback3[i].strip()}\n\n"
                             "Based on the feedback, is the proposed answer True or False? "
                             "Please respond clearly with 'True' or 'False'.")
        final_prompts.append(combined_prompt)

    final_responses = []
    final_probs = []
    for i in tqdm(range(0, len(final_prompts), batch_size)):
        batch_prompts = final_prompts[i:i+batch_size]
        batch_outputs = lm_utils.llm_response(batch_prompts, model_name, probs=True, max_new_tokens=10)
        batch_generated_texts = batch_outputs["generated_texts"]  # List[str] (batch_size, )
        batch_token_probs = batch_outputs["token_probs"]  # List[Dict[str, float]] (batch_size, (seq_len, ))
        final_responses.extend(batch_generated_texts)
        final_probs.extend(batch_token_probs)

    abstain_flags = []
    abstain_scores = []
    for resp, probs_dict in zip(final_responses, final_probs):
        predicted_label = lm_utils.answer_parsing(resp, model_name)
        if predicted_label == "A":
            abstain_flags.append(0)
        elif predicted_label == "B":
            abstain_flags.append(1)
        else:
            abstain_flags.append(random.randint(0, 1))
        found_score = 0.5
        if probs_dict is not None:
            prob_true = None
            prob_false = None
            for k, pval in probs_dict.items():
                norm_k = k.strip().lower()
                if norm_k == "true":
                    prob_true = pval
                elif norm_k == "false":
                    prob_false = pval
            if prob_true is not None or prob_false is not None:
                if predicted_label == "A":
                    found_score = 1 - prob_true
                elif predicted_label == "B":
                    found_score = prob_false
        abstain_scores.append(found_score)

    # Save feedback to a file if specified
    if feedback_out:
        feedback_data = []
        for idx, instance in enumerate(data["test"]):
            q_prompt = f"Question: {instance['question']}\n"
            for ck, ct in instance["choices"].items():
                q_prompt += f"{ck}: {ct}\n"
            feedback_data.append({
                "question": q_prompt,
                "proposed_answer": answers_given[idx],
                "feedbacks": [feedback1[idx], feedback2[idx], feedback3[idx]],
                "abstain_flag": abstain_flags[idx],
                "correct_flag": correct_flags[idx]
            })
        feedback_dir = f"feedbacks/{dataset_name}/multirelated"
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_path = f"{feedback_dir}/{model_name}_{dataset_name}_{source_language}_multirelated.json"
        with open(feedback_path, "w", encoding="utf-8") as ff:
            json.dump(feedback_data, ff, indent=4, ensure_ascii=False)
        print(f"[Saved feedbacks to {feedback_path}]")

    # Save predictions to a file if specified
    if local_out:
        out_data = {
            "correct_flags": correct_flags,
            "abstain_flags": abstain_flags,
            "abstain_scores": abstain_scores
        }
        out_dir = f"preds/{dataset_name}/multirelated"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{model_name}_{dataset_name}_{source_language}_multirelated.json"
        with open(out_path, "w", encoding="utf-8") as ff:
            json.dump(out_data, ff, indent=2, ensure_ascii=False)
        print(f"[Local output saved to {out_path}]")

    print("-" * 10, "Multi-Related", "-" * 10)
    print("Model:", model_name)
    print("Dataset:", dataset_name)
    print("Language:", source_language)
    final_scores = metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores)
    print("Metrics:", final_scores)

    if result_out:
        result_dir = f"results/{dataset_name}/multirelated"
        os.makedirs(result_dir, exist_ok=True)
        result_path = f"{result_dir}/{model_name}_{dataset_name}_{source_language}_multirelated.json"
        with open(result_path, "w", encoding="utf-8") as rf:
            json.dump(final_scores, rf, indent=2, ensure_ascii=False)
        print(f"[Saved result metrics to {result_path}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Which language model to use.")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset to run on (mmlu, hellaswag, belebele, etc.).")
    parser.add_argument("-s", "--speak", default="bn", help="Primary language code, e.g. 'bn', 'es', 'nl', etc.")
    parser.add_argument("-o", "--portion", default=1.0, type=float, help="Only use this fraction of dataset.")
    parser.add_argument("-l", "--local", default=False, action='store_true', help="If set, save local JSON of predictions.")
    parser.add_argument("-f", "--feedback", default=False, action='store_true', help="If set, save a separate file of generated feedback.")
    parser.add_argument("-r", "--result", default=False, action='store_true', help="If set, save result metrics to a local JSON file.")
    parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for generation.")
    parser.add_argument("--test-all", action="store_true", help="If specified, run all test languages (bn, kn, ml, mr, ne, ta, te) sequentially.")
    args = parser.parse_args()
    print("Arguments:", args)
    
    start_time = time.time()

    # init model
    lm_utils.llm_init(args.model)

    if args.test_all:
        test_languages = ["bn", "kn", "ml", "mr", "ne", "ta", "te"]
        for lang in test_languages:
            print(f"\n===== Running test for language: {lang} =====")
            run_pipeline(args.model, args.dataset, lang, args.portion, args.local, args.feedback, args.result, args.batch_size)
    else:
        run_pipeline(args.model, args.dataset, args.speak, args.portion, args.local, args.feedback, args.result, args.batch_size)

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
