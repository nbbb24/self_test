mooncake_master --port 50088 \
    --eviction_high_watermark_ratio 0.8 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000 \
    --enable_offload=false \
    --client_ttl=120
