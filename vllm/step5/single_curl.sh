#!/bin/bash
curl -s http://10.208.128.50:19401/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step5",
    "messages": [
      {
        "role": "user",
        "content": "Who are you? Where are you from?"
      }
    ],
    "max_tokens": 1024,
    "temperature": 1,
    "top_p": 0.95,
    "stream": false
  }'