# mulitdominationGame
def mulitdominationGame(n, k, p, time):
    
    alpha = 2
    beta = 1
    import numpy as np
    import random

    def generate(n):
        
        # Rewire the dedge
        def rewire(t):
            if random.random() < p:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                while adj[x, y] == 1 or x == y or adj[y, x]:
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                adj[x, y] = 1
                adj[i, t] = 0
            else:
                adj[i, t % n] = 1
                
        # construct the adjacent matrix
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(int(k / 2)):
                adj[i,(i + j + 1) % n] = 1
                
        # Rewire
        for i in range(n):
            for j in range(int(k / 2)):
                rewire((i + j + 1) % n)

        adj = adj + adj.T
        # join the diagonal elements
        for i in range(n):
            adj[i, i] = 1
            
        return adj
    
    # determine the degree
    def degreeIsZero(adj):
        for i in range(len(adj)):
            if sum(adj[i]) - 1 == 0:
                return True
        return False
    
    # K set
    def KD(n):
        return np.random.randint(1, high = 4, size = n)
    
    # c strategy
    def C(n):
        s=[]
        for i in range(n):
            s.append(random.randrange(0,2))
        return np.array(s)
    
    # v function
    def V(i):
        vl = adj[i].nonzero()
        s = 0
        for j in range(len(vl[0])):
            s += c[vl[0][j]]
        return s
    
    # return the sigma of the player i
    def G(i):
        gl = adj[i].nonzero()
        g = 0
        for j in range(len(gl[0])):
            if V(gl[0][j]) <= kd[gl[0][j]]:
                g += alpha
        return g
    
    # utility function
    def U(i):
        return G(i) - beta

    total_time = 0
    total_count = 0
    
    for x in range(time):

        adj = generate(n)
        while degreeIsZero(adj):
            adj = generate(n)
            
        # generate K
        kd = KD(n)
        
        # determine the adjacent matrix and K is legal
        t=0
        while t==0:
            f = 0
            for i in range(n):
                if kd[i] > sum(adj[i]) - 1:
                    kd = KD(n)
                    adj = generate(n)
                    f = 1
                    break
            if f == 0:
                t = 1    
                
        # generate C
        c = C(n)

        # play Game
        move_count = 0
        player_list =  []
        final = 0
        m = 0

        # if no player can action, then out of the while
        while(m != 30):

            # reset player list when all player action once
            if len(player_list) == 30:
                player_list =  []

            # randomly select one player to action
            player = random.randint(0, n-1)
            while player in player_list:
                player = random.randint(0, n-1)
            player_list.append(player)
            
            # #best response method
            # # the neighbor of player
            # neighbor = adj[player].nonzero()

            # # reset the player to action 0 for him choosing the best response
            # if c[player] == 1:
            #     c[player] = 0
            #     # do best response
            #     nonbest = 1
            #     for j in range(len(neighbor[0])):
            #         if V(neighbor[0][j]) < kd[neighbor[0][j]]:
            #             c[player] = 1
            #             nonbest = 0
                        
            #             break
            #     if nonbest:
            #         c[player] = 0
            #         move_count += 1
            # else:
            #     # do best response
            #     nonbest = 1
            #     for j in range(len(neighbor[0])):
            #         if V(neighbor[0][j]) < kd[neighbor[0][j]]:
            #             c[player] = 1
            #             nonbest = 0
            #             move_count += 1
            #             break

            #     if nonbest:
            #         c[player] = 0
            # #best response method
    
            # utility method
            # let the strategu of player be 1
            if c[player] == 1:
                if U(player) <= 0:
                    c[player] = 0
                    move_count += 1

            if c[player] == 0:

                c[player] = 1
                if U(player) > 0:
                    move_count +=1
                else:
                    c[player] = 0
            # utility method
            
            # count the number of elements in the dominating set
            sumc = sum(c)

            # if someone can action
            if final != sumc:
                final = sumc
                m = 0
            else:
                m += 1

            # none of the players can action
            if m == 30:
                break

                
        total_count += sum(c)
        total_time += move_count

    # result
    print('Average moves: '+str(float('{:.02f}'.format((total_time / time ) / n)))+'  Average size per node: '+str(float('{:.02f}'.format(total_count / time))))   

def symmetric(n, k, p, time):
    
    import numpy as np
    import random
    
    alpha = 2
    beta = 1 
    gamma = n * alpha + 1
    
    # generate the adjacent matrix
    def generate(n):
        
        # rewire edges
        def rewire(t):
            if random.random() < p:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                while adj[x, y] == 1 or x == y or adj[y, x] == 1:
                    x = random.randint(0,n-1)
                    y = random.randint(0,n-1)
                adj[x, y] = 1
                adj[i, t] = 0
            else:
                adj[i, t % n] = 1
                
        # init matrix
        adj = np.zeros((n, n))
        
        # WS model
        for i in range(n):
            for j in range(int(k / 2)):
                adj[i,(i + j + 1) % n] = 1
        
        # rewire
        for i in range(n):
            for j in range(int(k / 2)):
                rewire((i + j + 1) % n)
        
        # symmetric
        adj = adj + adj.T
        
        # join the diagonal elements
        for i in range(n):
            adj[i, i] = 1

        return adj
    
    # determine if degree is equal to zero
    def degreeIsZero(adj):
        for i in range(len(adj)):
            if sum(adj[i]) - 1 == 0:
                return True
        return False
    
    # c set
    def C(n):
        return np.zeros(30, dtype = 'int64')
    
    # v function
    def V(i):
        vl = adj[i].nonzero()
        s = 0
        for j in range(len(vl[0])):
            s += c[vl[0][j]]
        return s
    
    # g function
    def G(i):
        gl = adj[i].nonzero()
        s = 0
        for j in range(len(gl[0])):
            if V(gl[0][j]) == 1:
                s += alpha
        return s
    
    # w funciotn
    def W(i):
        wl = adj[i].nonzero()
        s = 0
        for j in range(len(wl[0])):
            if wl[0][j] != i:
                s += c[wl[0][j]] * c[i] * gamma
        return s
    
    # utility funciotn
    def U(i):
        return G(i) - beta - W(i)
    
    total_time = 0
    total_count = 0
    
    for x in range(time):
        
        # generate graph and some parameter
        adj = generate(n)
        while degreeIsZero(adj):
            adj = generate(n)
        c = C(n)
        move_count = 0
        player_list =  []
        final = 0
        m = 0
        
        # game
        while(m != 30):
            
            # randomly choose some player
            if len(player_list) == 30:
                player_list = []

            player = random.randint(0, n-1)
            while player in player_list:
                player = random.randint(0, n-1)
            player_list.append(player)

            # determine player's action
            
            if c[player] == 1:

                if U(player) > 0:
                    pass
                else:
                    c[player] = 0
                    move_count += 1
            else:
                c[player] = 1
                if U(player) <= 0:
                    c[player] = 0
                else:
                    move_count += 1
            
            
            # determine if out of loop
            sumc = sum(c)
            
            if final != sumc:
                final = sumc
                m = 0
            else:
                m += 1
            if m == 30:
                break

        total_count += sum(c)
        total_time += move_count
    print('Average moves: '+str(float('{:.02f}'.format((total_time / time ) / n)))+'  Average size per node: '+str(float('{:.02f}'.format(total_count / time))))    

def matching(n, k, p, times):
    import numpy as np
    import random
    
    # generate graph
    def generate(n):
        
        # rewire edges
        def rewire(t):
            if random.random() < p:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                while adj[x, y] == 1 or x == y or adj[y, x]:
                    x = random.randint(0, n - 1)
                    y = random.randint(0, n - 1)
                adj[x, y] = 1
                adj[i, t] = 0
            else:
                adj[i, t] = 1
                
        # generate graph
        adj = np.zeros((n, n))

        for i in range(n):
            for j in range(int(k / 2)):
                adj[i, (i + j + 1) % n] = 1

        for i in range(n):
            for j in range(int(k / 2)):
                rewire((i + j + 1) % n)
        
        # symmetric
        adj = adj + adj.T

        return adj
    
    # c set
    def C(n):
        return np.zeros(30) - 1 
    
    # degree
    def degree(adj):
        return sum(adj)
    
    # determine if some is zero
    def degreeIsZero(adj):
        for i in range(len(adj)):
            if sum(adj[i]) == 0:
                return True
        return False
    
    # utility function
    def U(i):
        ul = adj[i].nonzero()[0]
        ud = 0
        ui = -1
        for k in range(len(ul)):
            if 1 / degreeOfadj[ul[k]] > ud:
                if c[ul[k]] == -1:
                    ud = 1 / degreeOfadj[ul[k]]
                    ui = ul[k]
        return ud, ui
    
    # player order
    def playlist(adj):
        plist = []
        md = 1
        while len(plist) != n:
            mdarray = np.where(degreeOfadj == md)[0]
            for mdi in range(len(mdarray)):
                plist.append(mdarray[mdi])
            md += 1
        return np.array(plist) 
    
    
    total_time = 0
    total_count = 0

    for x in range(times):
        
        adj = generate(n)
        while degreeIsZero(adj):
            adj = generate(n)

        degreeOfadj = degree(adj)

        c = C(n)
        
        play_list = playlist(adj)
        
        # play game
        move_count = 0
        for order in range(len(play_list)):
            player = play_list[order]
            if c[player] == -1:
                if U(player)[0]>0:
                    c[player] = U(player)[1]
                    c[U(player)[1]] = player
                    move_count += 1
                    
        total_time += move_count
        total_count += (n-sum(c==-1))/2
        
    print('Average moves: '+str(float('{:.01f}'.format((total_time) / times)))+'  Average size per node: '+str(float('{:.01f}'.format(total_count / times))))
    
print("Input what you want to play, mulit, symm, match: ")
s = input()
while s!= 'mulit' and s!='symm' and s!='match':
    print('Wrong Input. Restart input one of them: ')
    s = input()
if s=='mulit':
    for pr in range(0,10,2):
        mulitdominationGame(30, 4, pr/10, 100)
        
elif s=='symm':
    for pr in range(0,10,2):
        symmetric(30, 4, pr/10, 100)
        
elif s=='match': 
    for pr in range(0,10,2):
        matching(30, 4, pr/10, 100)