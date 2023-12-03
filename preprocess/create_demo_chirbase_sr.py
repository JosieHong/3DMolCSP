import pickle

root_path = "./data/ChirBase/exp/chirbase_sr_clean_train.pkl"
out_path = root_path.replace('.pkl', '_demo.pkl')
with open(root_path, 'rb') as file: 
	data = pickle.load(file)
	print('Load {} data from {}'.format(len(data), root_path))
with open(out_path, 'wb') as f: 
	pickle.dump(data[:1000], f)
	print('Save {}'.format(out_path))

root_path = "./data/ChirBase/exp/chirbase_sr_clean_test.pkl"
out_path = root_path.replace('.pkl', '_demo.pkl')
with open(root_path, 'rb') as file: 
	data = pickle.load(file)
	print('Load {} data from {}'.format(len(data), root_path))
with open(out_path, 'wb') as f: 
	pickle.dump(data[:100], f)
	print('Save {}'.format(out_path))