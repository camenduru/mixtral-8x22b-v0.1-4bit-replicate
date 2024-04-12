from cog import BasePredictor, Input
import os
import time
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "mistral-community/Mixtral-8x22B-v0.1-4bit"
MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/mistral-community/Mixtral-8x22B-v0.1-4bit/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        #Download with Pget
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE
        )

    def predict(
        self,
        prompt: str = Input("What are the 20 countries with the largest population?"),
        max_new_tokens: int = Input(default=128),
        repetition_penalty: float = Input(default=2.0),
        length_penalty: float = Input(default=1.0),
        num_beams: int = Input(default=1),
        do_sample: bool = True,
        temperature: float = Input(default=0.2),
        top_k: int = Input(default=0),
        top_p: float = Input(default=0.8),
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)