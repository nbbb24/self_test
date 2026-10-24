import torch._dynamo.trace_rules as rules
from torch._dynamo.variables.functions import SkipFunctionVariable
for rule_map in rules.torch_name_rule_map:
    for key in list(rule_map):
        if key.rsplit('.', 1)[-1] in {'manual_seed', 'manual_seed_all', 'get_rng_state', 'set_rng_state'}:
            rule_map[key] = SkipFunctionVariable
rules.get_torch_obj_rule_map.cache_clear()
