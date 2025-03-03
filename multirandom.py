#!/usr/bin/env python3

import json
import argparse
import random
import lm_utils
import metrics
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Which language model to use.")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset to run on (mmlu, hellaswag, belebele, etc.).")
    parser.add_argument("-s", "--speak", required=True, help="Primary language code, e.g. 'bn', 'es', 'nl', etc.")
    parser.add_argument("-o", "--portion", default=1.0, type=float, help="Only use this fraction of dataset.")
    parser.add_argument("-l", "--local", default=False, action='store_true', help="If set, save local JSON of predictions.")
    parser.add_argument("-f", "--feedback", default=False, action='store_true', help="If set, save a separate file of generated feedback.")
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    source_language = args.speak
    portion = args.portion
    local_out = args.local
    feedback_out = args.feedback


    language_list = ["English", "Russian", "German", "Chinese", "French", "Spanish", "Italian", "Dutch", "Vietnamese",
                     "Indonesian", "Arabic", "Hungarian", "Romanian", "Danish", "Slovak", "Ukrainian", "Catalan", "Serbian",
                     "Croatian", "Hindi", "Bengali", "Tamil", "Nepali", "Malayalam", "Marathi", "Telugu", "Kannada"]

    # Initialize model
    lm_utils.llm_init(model_name)

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
    for instance in tqdm(data["test"]):
        prompt_text = f"Question: {instance['question']}\n"
        for choice_key, choice_text in instance["choices"].items():
            prompt_text += f"{choice_key}: {choice_text}\n"
        prompt_text += "Choose one answer from the above choices. Just provide the letter (A, B, C, or D) of the correct choice. The answer is"
        answer_text = lm_utils.llm_response(
            prompt_text, model_name,
            probs=False,
            max_new_tokens=10
        )

        label = lm_utils.answer_parsing(answer_text, model_name)
        if label == instance["answer"]:
            correct_flags.append(1)
        else:
            correct_flags.append(0)

        answers_given.append(answer_text)


    print("\n2: Generating multilingual feedback for each answer.")
    feedback1, feedback2, feedback3 = [], [], []
    for i, instance in enumerate(tqdm(data["test"])):
        base_prompt = f"Question: {instance['question']}\n"
        for ck, ct in instance["choices"].items():
            base_prompt += (ck + ": " + ct + "\n")
        base_prompt += (
            f"Choose one answer from the above choices. Proposed answer: {answers_given[i].strip()}\n"
            "Please review the proposed answer and provide a paragraph of feedback on its correctness."
            " Feedback should be in <LANG>.\n"
            "Feedback:"
        )

        fbs = []
        for _ in range(3):
            lang_prompt = base_prompt.replace("<LANG>", random.choice(language_list))
            feedback_resp = lm_utils.llm_response(
                lang_prompt,
                model_name,
                probs=False,
                temperature=1
            )
            if not feedback_resp.strip():
                feedback_resp = "No feedback provided."
            fbs.append(feedback_resp.split("\n")[0].strip())

        feedback1.append(fbs[0])
        feedback2.append(fbs[1])
        feedback3.append(fbs[2])

    print("\n3: Make abstain decision based on feedback.")
    abstain_flags = []
    abstain_scores = []

    final_prompts = []
    for i, instance in enumerate(data["test"]):
        combined_prompt = (
            f"Question: {instance['question']}\n"
        )
        for ck, ct in instance["choices"].items():
            combined_prompt += f"{ck}: {ct}\n"

        combined_prompt += (
            f"Choose one answer. Proposed answer: {answers_given[i].strip()}\n\n"
            f"Feedback 1: {feedback1[i].strip()}\n\n"
            f"Feedback 2: {feedback2[i].strip()}\n\n"
            f"Feedback 3: {feedback3[i].strip()}\n\n"
            "Based on the feedback, is the proposed answer True or False? Please respond clearly with 'True' or 'False'."
        )
        final_prompts.append(combined_prompt)

    # Now call the LLM again for each final prompt, parse True/False
    for final_prompt in tqdm(final_prompts):
        resp, probs_dict = lm_utils.llm_response(
            final_prompt,
            model_name,
            probs=True,
            max_new_tokens=10
        )
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
            if prob_true is not None and prob_false is not None:
                if predicted_label == "True" and prob_true is not None:
                    found_score = 1 - prob_true
                elif predicted_label == "False" and prob_false is not None:
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

        path_feedback = f"feedbacks/{model_name}_{dataset_name}_{source_language}_multirandom.json"
        with open(path_feedback, "w", encoding="utf-8") as ff:
            json.dump(feedback_data, ff, indent=4, ensure_ascii=False)
        print(f"[Saved feedbacks to {path_feedback}]")

    # Save predictions to a file if specified
    if local_out:
        out_data = {
            "correct_flags": correct_flags,
            "abstain_flags": abstain_flags,
            "abstain_scores": abstain_scores
        }
        out_path = f"preds/{model_name}_{dataset_name}_{source_language}_multirandom.json"
        with open(out_path, "w", encoding="utf-8") as ff:
            json.dump(out_data, ff, indent=2, ensure_ascii=False)
        print(f"[Local output saved to {out_path}]")


    print("-"*10, "Multi-Random", "-"*10)
    print("Model:", model_name)
    print("Dataset:", dataset_name)
    print("Language:", source_language)
    final_scores = metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores)
    print("Metrics:", final_scores)

if __name__ == "__main__":
    main()