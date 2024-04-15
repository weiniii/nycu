import gzip
import numpy as np
from numba import jit

train_images_file_path = 'train-images-idx3-ubyte.gz'
train_labels_file_path = 'train-labels-idx1-ubyte.gz'

bin_size = 128
image_width = 28
image_height = 28
image_size = image_width * image_height
image_offset = 16
label_offset = 8

def get(image_file_path, label_file_path):
    
    file = gzip.open(image_file_path, 'rb')
    file_read = file.read(image_offset)
    
    buff = file.read()
    
    image_data = np.frombuffer(buff, dtype=np.uint8)
    image_data = np.reshape(image_data, (-1, 28, 28)).astype(int)
    
    file.close()
    
    print(f"Found {(len(buff) / image_size) :.0f} images")
    
    file = gzip.open(label_file_path, 'rb')
    file_read = file.read(label_offset)
    
    buff = file.read()
    
    label_data = np.frombuffer(buff, dtype=np.uint8)
    label_data = np.array(label_data).astype(int)
    
    file.close() 
    
    print(f"Found {(len(buff)) :.0f} labels")
    
    del file_read
    del buff
    
    image_data = image_data / bin_size
    image_data = image_data.astype(int)
    
    return image_data, label_data

@jit
def E_step(X, image_width, image_height, probability, N, Lambda):
    new_responsibility = np.zeros((N, 10))

    for image_num in range(N):
        for class_num in range(10):
            new_responsibility[image_num, class_num] = Lambda[class_num]
            
            for width_pixel in range(image_width):
                for height_pixel in range(image_height):
                    if X[image_num, width_pixel, height_pixel]:
                        new_responsibility[image_num, class_num] *= probability[class_num, width_pixel, height_pixel]
                    else:
                        new_responsibility[image_num, class_num] *= (1.0 - probability[class_num, width_pixel, height_pixel])

        summation = np.sum(new_responsibility[image_num, :])
        if summation:
            new_responsibility[image_num, :] /= summation
    
    return new_responsibility

@jit
def M_step(X, image_width, image_height, responsibility, N):
    
    sum_of_responsibility = np.zeros(10)
    for class_num in range(10):
        sum_of_responsibility[class_num] += np.sum(responsibility[:, class_num])

    new_probability = np.zeros((10, image_width, image_height))
    lam = np.zeros(10)

    for class_num in range(10):
        for width_pixel in range(image_width):
            for height_pixel in range(image_height):
                for image_num in range(N):
                    new_probability[class_num, width_pixel, height_pixel] += responsibility[image_num, class_num] * X[image_num, width_pixel, height_pixel]
                    
                new_probability[class_num, width_pixel, height_pixel] = (new_probability[class_num, width_pixel, height_pixel] + 1e-9) / (
                        sum_of_responsibility[class_num] + 1e-9 * image_width * image_height)
        
        lam[class_num] = (sum_of_responsibility[class_num] + 1e-9) / (np.sum(sum_of_responsibility) + 1e-9 * 10)

    return lam, new_probability

def show(probability, diff, count, image_width, image_height):
    
    imagination = (probability >= 0.5)

    print('')
    for class_num in range(10):
        print(f'class {class_num}:')
        for row_num in range(image_width):
            for col_num in range(image_height):
                print(f'\033[93m1\033[00m', end=' ') if imagination[class_num, row_num, col_num] else print('0', end=' ')
            print('')
        print('')
    
    print(f'No. of Iteration: {count}, Difference: {diff}')

def show_result(matching, probability):
    
    imagination = (probability >= 0.5)
    for class_num in range(10):
        print(f'labeled class {class_num}:')
        ind = matching.index(class_num)
        for row_num in range(image_width):
            for col_num in range(image_height):
                print(f'\033[93m1\033[00m', end=' ') if imagination[ind, row_num, col_num] else print('0', end=' ')
            print('')
        print('')
    

def statistic(responsibility, Y, N):
    count = np.zeros((10, 10)).astype(int)
    for i in range(N):
        unknow_class = np.argmax(responsibility[i])
        count[unknow_class, Y[i]] += 1
    
    return count

def match(count):
    matching = [-1 for _ in range(10)] 

    for _ in range(10):

        idx = np.unravel_index(np.argmax(count), (10, 10))

        matching[idx[0]] = idx[1]

        for k in range(10):
            count[idx[0]][k] = -1
            count[k][idx[1]] = -1
        
    return matching

def final_confusion(i, tp, fn, fp, tn, t, f):
    pt = ['', f'Predict number {i}', f'Predict not number {i}']
    p1 = [f'Is number {i}', tp, fn]
    p2 = [f'Isn\'t number {i}', fp, tn]
    
    print(f'Confusion Matrix {i}:')
    print(f'{pt[0]:<18}  {pt[1]:^20}  {pt[2]:^20}')
    print(f'{p1[0]:<18}  {p1[1]:^20}  {p1[2]:^20}')
    print(f'{p2[0]:<18}  {p2[1]:^20}  {p2[2]:^20}')
    print('')
    print(f'Sensitivity (Successfully predict number {i})     : {tp / t:.7f}')
    print(f'Specificity (Successfully predict not number {i}) : {tn / f:.7f}')
    
    print('')
    print('-' * 60)


X, Y = get(train_images_file_path, train_labels_file_path)
N = len(X)

Lambda = np.full(10, 0.1)
probability = np.random.uniform(0.0, 1.0, (10, image_width, image_height))

for class_num in range(10):
    probability[class_num, :] /= np.sum(probability[class_num, :])
responsibility = np.zeros((N, 10))

iteration = 0

while True:
    
    old_probability = probability
    iteration += 1

    responsibility = E_step(X, image_width, image_height, probability, N, Lambda)

    Lambda, probability = M_step(X, image_width, image_height, responsibility, N)
    
    diff = np.linalg.norm(probability - old_probability)
    
    
    show(probability, diff, iteration, image_width, image_height)
    
    if diff < 0.15 or iteration >= 25:
        break
    
responsibility = E_step(X, image_width, image_height, probability, N, Lambda)
count = statistic(responsibility, Y, N)
matching = match(count)

show_result(matching, probability)

count = statistic(responsibility, Y, N)

s = 0
for i in range(10):
    
    true_y = np.sum(Y == i)
    false_y = N - true_y
    
    true_positive = count[matching.index(i)][i]
    false_negative = true_y - true_positive
    
    pred_y = np.sum(count[matching.index(i)])
    
    false_positive = pred_y - true_positive
    true_negative = false_y - false_positive
    
    s += false_negative
    
    final_confusion(i, true_positive, false_negative, false_positive, true_negative, true_y, false_y)
    

print(f'Total iteration to converge: {iteration}')
print(f'Total error rate: {s / N : .10f}')

