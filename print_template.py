import sys
import json
sys.path.insert(0, '/home/ramji.purwar/DecompGTI/DecompGTI/GraphInstruct/LLaMAFactory/src')
from llamafactory.data.template import TEMPLATES
t = TEMPLATES['default']
res = {
    'system': t.format_system.slots,
    'user': t.format_user.slots,
    'assistant': t.format_assistant.slots
}
with open('debug_template.json', 'w') as f:
    json.dump(res, f)
