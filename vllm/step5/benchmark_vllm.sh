mkdir result
vllm bench serve \
  --backend openai \
  --base-url http://10.208.128.50:19289 \
  --endpoint /v1/chat/completions \
  --model step5 \
  --tokenizer /model/target \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 122880 \
  --prefix-repetition-suffix-len 960 \
  --prefix-repetition-output-len 32 \
  --prefix-repetition-num-prefixes 1 \
  --num-prompts 16 \
  --max-concurrency 16 \
  --disable-tqdm --ignore-eos --seed 1000 \
  --save-result --result-dir ./result/pool_test_1106