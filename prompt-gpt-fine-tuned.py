from transformers import pipeline
import time

GPT_FINE_TUNED_FILE_01 = "fine_tuned_models/gpt-neo-125M-ML-ArXiv-Papers-01"
GPT_FINE_TUNED_FILE_02 = "fine_tuned_models/gpt-neo-125M-ML-ArXiv-Papers-02"

prompt = "a general framework of semi-supervised dimensionality reduction for"

start = time.time()
print("Time elapsed on working...")

generator1 = pipeline('text-generation', model=GPT_FINE_TUNED_FILE_01)
text1 = generator1(prompt,
                 min_length=20,
                 max_length=100,
                 #do_sample=True,
                 temperature=0.95,
                 pad_token_id=50256,
                 num_return_sequences=1, )

generator2 = pipeline('text-generation', model=GPT_FINE_TUNED_FILE_02)
text2 = generator2(prompt,
                 min_length=20,
                 max_length=100,
                 #do_sample=True,
                 temperature=0.95,
                 pad_token_id=50256,
                 num_return_sequences=1, )

generator_neo_125M = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
text_neo_125M = generator_neo_125M(prompt,
                 min_length=20,
                 max_length=100,
                 #do_sample=True,
                 temperature=0.95,
                 pad_token_id=50256,
                 num_return_sequences=1, )

generator_bloom_560m = pipeline('text-generation', model='bigscience/bloom-560m')
text_bloom_560m = generator_bloom_560m(prompt,
                 min_length=20,
                 max_length=100,
                 #do_sample=True,
                 temperature=0.95,
                 pad_token_id=50256,
                 num_return_sequences=1, )

print('GPT_FINE_TUNED_FILE_01', text1)
print('GPT_FINE_TUNED_FILE_02', text2)
print('EleutherAI/gpt-neo-125M', text_neo_125M)
print('bigscience/bloom-560m', text_bloom_560m)

time.sleep(0.01)
end = time.time()
print("Time consumed in working: ", end - start)
