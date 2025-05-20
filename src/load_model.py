from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# Start timer
start_time = datetime.now()

model_name = "Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loaded successfully.")

# prepare the model input
prompt = """You are tasked with solving an ARC-AGI reasoning challenge. CRITICALLY IMPORTANT: Your response must ONLY contain a list of matrices representing your solution.

DO NOT include ANY explanations, reasoning, or text after you stop thinking.
DO NOT describe your approach or explain your solution after you stop thinking.
ONLY output the solution in the format [[a,b,c], [d,e,f], [g,h,i]] where the values represent your answer.

For example, if your solution is a 3x3 grid, your entire response should look like:
[[1,2,3],[4,5,6],[7,8,9]]

NOTHING ELSE. No text before or after. ONLY the list of lists with your solution."""

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

# End timer
end_time = datetime.now()
print(f"\nTime taken: {end_time - start_time}\n")