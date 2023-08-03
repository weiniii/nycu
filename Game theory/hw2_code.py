def fictitious_play(Q):
    import numpy as np
    
    def PureNE(arr1, arr2):
        p1 = []
        p2 = []
        if arr1[0, 0] >= arr1[1, 0] and arr2[0, 0] >= arr2[1, 0]:
            p1.append('R1')
            p2.append('C1')
        if arr1[1, 0] >= arr1[0, 0] and arr2[0, 1] >= arr2[1, 1]:
            p1.append('R2')
            p2.append('C1')
        if arr1[0, 1] >= arr1[1, 1] and arr2[1, 0] >= arr2[0, 0]:
            p1.append('R1')
            p2.append('C2')
        if arr1[1, 1] >= arr1[0, 1] and arr2[1, 1] >= arr2[0, 1]:
            p1.append('R2')
            p2.append('C2')
        return p1,p2

    count_p1 = np.zeros(2)
    count_p2 = np.zeros(2)

    if np.random.rand(1)[0] < 0.5:
        count_p1[0] += 1
        p1_now = 'R1'
    else:
        count_p1[1] += 1
        p1_now = 'R2'
        
    if np.random.rand(1)[0] < 0.5:
        count_p2[0] += 1
        p2_now = 'C1'
    else:
        count_p2[1] += 1
        p2_now = 'C2'

    p1_old = p1_now
    p2_old = p2_now
    print('Init strategy: '+str(p1_now)+','+str(p2_now)+' by random')
    match Q:
        case 'Q1':
            utility_p1 = np.array([[-1, 1], [0, 3]])
            utility_p2 = np.array([[-1, 1], [0, 3]])
        case 'Q2':
            utility_p1 = np.array([[2, 1], [0, 3]])
            utility_p2 = np.array([[2, 1], [0, 3]])
        case 'Q3':
            utility_p1 = np.array([[1, 0], [0, 0]])
            utility_p2 = np.array([[1, 0], [0, 0]])
        case 'Q4':
            utility_p1 = np.array([[0, 2], [2, 0]])
            utility_p2 = np.array([[1, 0], [0, 4]])
        case 'Q5':
            utility_p1 = np.array([[0, 1], [1, 0]])
            utility_p2 = np.array([[1, 0], [0, 1]])
        case 'Q6':
            utility_p1 = np.array([[10, 0], [0, 10]])
            utility_p2 = np.array([[10, 0], [0, 10]])
        case 'Q7':
            utility_p1 = np.array([[0, 1], [1, 0]])
            utility_p2 = np.array([[0, 1], [1, 0]])
        case 'Q8':
            utility_p1 = np.array([[3, 0], [0, 2]])
            utility_p2 = np.array([[2, 0], [0, 3]])
        case 'Q9':
            utility_p1 = np.array([[3, 0], [2, 1]])
            utility_p2 = np.array([[3, 0], [2, 1]])
            
    p1,p2 = PureNE(utility_p1, utility_p2)
    print(str(Q)+' result: ')
    if len(p1)!=0:
        print('  All Pure-strategy Nash equilibrium:')
        for i in range(len(p1)):
            print(' ',p1[i],p2[i])
        out = 1
        while(out):
            probability_p1 = np.array([count_p1[0]/sum(count_p1), count_p1[1]/sum(count_p1)])
            probability_p2 = np.array([count_p2[0]/sum(count_p2), count_p2[1]/sum(count_p2)])

            if probability_p2[0] * utility_p1[0, 0] + probability_p2[1] * utility_p1[0, 1] > probability_p2[0] * utility_p1[1, 0] + probability_p2[1] * utility_p1[1, 1]:
                p1_now = 'R1'
                count_p1[0] += 1
            elif probability_p2[0] * utility_p1[0, 0] + probability_p2[1] * utility_p1[0, 1] == probability_p2[0] * utility_p1[1, 0] + probability_p2[1] * utility_p1[1, 1]:
                if np.random.rand(1)[0] < 0.5:
                    p1_now = 'R1'
                    count_p1[0] += 1
                else:
                    p1_now = 'R2'
                    count_p1[1] += 1
            else:
                p1_now = 'R2'
                count_p1[1] += 1
                
            if probability_p1[0] * utility_p2[0, 0] + probability_p1[1] * utility_p2[0, 1] > probability_p1[0] * utility_p2[1, 0] + probability_p1[1] * utility_p2[1, 1]:
                p2_now = 'C1'
                count_p2[0] += 1
            elif probability_p1[0] * utility_p2[0, 0] + probability_p1[1] * utility_p2[0, 1] == probability_p1[0] * utility_p2[1, 0] + probability_p1[1] * utility_p2[1, 1]:
                if np.random.rand(1)[0] < 0.5:
                    p2_now = 'C1'
                    count_p2[0] += 1
                else:
                    p2_now = 'C2'
                    count_p2[1] += 1
            else:
                p2_now = 'C2'
                count_p2[1] += 1
                
            for i in range(len(p1)):
                if p1[i] == p1_now and p2[i] == p2_now:
                    print('Found pure-strategy Nash Equilibrium: '+str(p1_now)+' , '+str(p2_now))
                    out = 0
                    break
    else:
        print('No pure-strategy Nash Equilibrium')
        if Q=='Q5':
            print(' best-reply path:')
        time = 10000
        round = 0
        for i in range(time):
            probability_p1 = np.array([count_p1[0]/sum(count_p1), count_p1[1]/sum(count_p1)])
            probability_p2 = np.array([count_p2[0]/sum(count_p2), count_p2[1]/sum(count_p2)])

            if probability_p2[0] * utility_p1[0, 0] + probability_p2[1] * utility_p1[0, 1] > probability_p2[0] * utility_p1[1, 0] + probability_p2[1] * utility_p1[1, 1]:
                p1_now = 'R1'
                count_p1[0] += 1
            elif probability_p2[0] * utility_p1[0, 0] + probability_p2[1] * utility_p1[0, 1] == probability_p2[0] * utility_p1[1, 0] + probability_p2[1] * utility_p1[1, 1]:
                if np.random.rand(1)[0] < 0.5:
                    p1_now = 'R1'
                    count_p1[0] += 1
                else:
                    p1_now = 'R2'
                    count_p1[1] += 1
            else:
                p1_now = 'R2'
                count_p1[1] += 1
                
            if probability_p1[0] * utility_p2[0, 0] + probability_p1[1] * utility_p2[0, 1] > probability_p1[0] * utility_p2[1, 0] + probability_p1[1] * utility_p2[1, 1]:
                p2_now = 'C1'
                count_p2[0] += 1
            elif probability_p1[0] * utility_p2[0, 0] + probability_p1[1] * utility_p2[0, 1] == probability_p1[0] * utility_p2[1, 0] + probability_p1[1] * utility_p2[1, 1]:
                if np.random.rand(1)[0] < 0.5:
                    p2_now = 'C1'
                    count_p2[0] += 1
                else:
                    p2_now = 'C2'
                    count_p2[1] += 1
            else:
                p2_now = 'C2'
                count_p2[1] += 1
                
            if round != 4 and Q=='Q5':
                if p1_old != p1_now or p2_old != p2_now:
                    print(' ',p1_old,p2_old)
                    p1_old = p1_now
                    p2_old = p2_now
                    round += 1
                
        print('Mix-strategy converge: '+str(probability_p1)+','+str(probability_p2)+' by '+str(time)+' times')

print('Input what you want to play (Q1~Q9):')
Q = input()
question = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9']
while Q not in question:
    print('Input error, please re-input: ')
    Q = input()
fictitious_play(Q)