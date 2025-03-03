import torch
from transformers import AutoTokenizer, AutoConfig
from enc_dec_model_runner import EncDecModelRunner

class LLM:
    def __init__(self, model, engine_dir, engine_name="enc_dec", debug_mode=False):
        self.model_name = model
        self.engine_dir = engine_dir
        self.engine_name = engine_name
        self.debug_mode = debug_mode
        self.runner = EncDecModelRunner.from_engine(self.engine_name,
                                                    self.engine_dir,
                                                    debug_mode=self.debug_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)

    def generate(self, prompts, max_new_tokens=64, return_dict=False, temperature=1.0, num_beams=1):
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids.to("cuda").type(torch.int32)

        decoder_start_token_id = self.config.decoder_start_token_id
        decoder_input_ids = torch.tensor([[decoder_start_token_id]]).to("cuda").type(torch.int32)
        decoder_input_ids = decoder_input_ids.repeat(input_ids.shape[0], 1)

        outputs = self.runner.generate(
            encoder_input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            debug_mode=self.debug_mode,
            return_dict=return_dict,
            temperature=temperature,
        )
        if isinstance(outputs, dict):
            output_ids = outputs["output_ids"][:, 0, :] # (batch_size, seq_len)
            log_probs = outputs["log_probs"][:, 0, :]
        else:
            output_ids = outputs[:, 0, :]

        generated_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        if isinstance(outputs, dict):
            token_probs_list = []
            for seq_ids, seq_log_probs in zip(output_ids, log_probs):
                token_probs = {}
                for token_id, log_prob in zip(seq_ids, seq_log_probs):
                    token_str = self.tokenizer.decode([token_id]).strip()
                    token_probs[token_str] = torch.exp(log_prob)
                token_probs_list.append(token_probs)
            return {"generated_texts": generated_texts, "token_probs": token_probs_list}
        
        return generated_texts


if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    llm = LLM(model="CohereForAI/aya-101", engine_dir="/data/aya-101-trt-bf16-engine/")
    outputs = llm.generate(prompts, return_dict=True)

    if isinstance(outputs, dict):
        for gen_text, tok_prob in zip(outputs["generated_texts"], outputs["token_probs"]):
            print(f"Generated text: {gen_text}")
            for tok, prob in tok_prob.items():
                print(f"Token: {tok}, Prob: {prob}")
    else:
        print(outputs)
