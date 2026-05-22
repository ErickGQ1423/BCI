import pickle
with open('/home/lab-admin/Documents/CurrentStudy/sub-S26CLASS_SUBJ_008/models/sub-S26CLASS_SUBJ_008_model.pkl', 'rb') as f:
    m = pickle.load(f)
print(type(m))
if isinstance(m, dict):
    print('Keys:', list(m.keys()))
    for k,v in m.items():
        print(f'  {k}: {type(v).__name__}', getattr(v, 'shape', ''))
else:
    print('Classes:', m.classes_)