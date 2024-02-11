import json
import jsonlines

f = open('regen_formatted.json')
data = json.load(f)
f.close()

with jsonlines.open('alpaca_hu.jsonl', mode='w') as writer:
    for d in data:
        writer.write({'instruction': d['instruction'], 'input': d['input'], 'output': d['output']})

with jsonlines.open('alpaca_hu_input.jsonl', mode='w') as writer:
    for d in data:
        writer.write({'input': (d['instruction']+'\n'+d['input']).strip(), 'output': d['output']})

with jsonlines.open('alpaca_hu_instruction.jsonl', mode='w') as writer:
    for d in data:
        writer.write({'instruction': (d['instruction']+'\n'+d['input']).strip(), 'output': d['output']})

with jsonlines.open('alpaca_hu_text.jsonl', mode='w') as writer:
    for d in data:
        writer.write({'text': 'Alább egy utasítás következik, ami egy feladatot ír le. Írj egy megfelelő választ, ami teljesíti a kérést!\n\n### Utasítás:\n'+(d['instruction']+'\n'+d['input']).strip()+'\n\n### Válasz:\n'+d['output']})

