from transformers import pipeline
import time

prompt = "gpt code architecture is"

start = time.time()
print("Time elapsed on working...")
# generator = pipeline('text-generation', model='bigscience/bloom-560m')
# generator = pipeline('text-generation', model='gpt2')
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
# generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

text = generator(prompt,
                 min_length=50,
                 max_length=200,
                 # do_sample=True,
                 temperature=0.95,
                 pad_token_id=50256,
                 num_return_sequences=1, )
print(text)
time.sleep(0.01)
end = time.time()
print("Time consumed in working: ", end - start)
