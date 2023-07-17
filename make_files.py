from processFile import Event
import numpy as np 
from tqdm import tqdm

train_dir = 'data/train/'
test_dir = 'data/test/'
solution_dir = 'data/solution/'
evt = Event('merged_tree_multPart_20230707.root', train_dir, test_dir, solution_dir)


train_start_stop_points = list(zip(np.arange(0, 50000, 1000), np.arange(1000, 51000, 1000))) 
test_start_stop_points = list(zip(np.arange(90000, 100000, 1000), np.arange(91000, 110000, 1000))) 
# about a minute to process 1000 events
for point in tqdm(train_start_stop_points): 
    evt.make_file_from_evts(point[0], point[1])
    evt.save_train_csv()

# make test files and solution files 
# counter = 0  
# for point in tqdm(test_start_stop_points):
#     evt.make_file_from_evts(point[0], point[1])
#     evt.save_test_csv()
#     if point[0] > 94000: 
#         is_private = False
#     else: 
#         is_private = True 
#     evt.save_solution_csv(counter, is_private) 
#     counter += 1

