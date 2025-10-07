import pickle
# Leggi actions
with open('output/explainability/actions_narrativo_bert-base-uncased_20251007_144056.pkl', 'rb') as f:
    actions = pickle.load(f)

print('Actions structure:')
print('Keys:', actions.keys())
if 'class_0' in actions:
    print(f'\nClass 0 actions: {len(actions["class_0"])}')
    for i, (action, stats) in enumerate(list(actions['class_0'].items())[:3]):
        print(f'{i+1}. {action[:60]}...')
        print(f'   Stats: {stats}')