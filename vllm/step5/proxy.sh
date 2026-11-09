unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

/opt/upgrade/.venv/bin/python -B \
  /home/sxy/script/load_balance_proxy_server_example.py \
  --host 10.208.128.50 --port 19401 \
  --prefiller-hosts 10.208.128.50 \
  --prefiller-ports 19289 \
  --decoder-hosts 10.208.128.46 10.208.128.47 10.208.128.48 10.208.128.49 \
  --decoder-ports 19390 19390 19390 19390 \
  --workers 1