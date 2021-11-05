
#//7,3 
#// set(1, 1+1), set(1, 1+2) set(1, 1+3) // 1100000 1010000 1001000 
#// set(2, 2+1), set(2, 2+2) set(2, 2+3) // 0110000 0101000 0100100
#// set(3, 3+1), set(3, 3+2) set(3, 3+3) // 0011000 0010100 0010010 
#// set(4, 4+1), set(4, 4+2) set(4, 4+3) // 0001100 0001010 0001001 
#// set(5, 5+1), set(5, 5+2) set(5, 5+3) // 0000110 0000101 1000100 
#// set(6, 6+1), set(6, 6+2) set(6, 6+3) // 0000011 1000010 0100010 
#// set(7, 7+1), set(7, 7+2) set(7, 7+3) // 1000001 0100001 0010001

def set(a,b, c, l): 
    s = ''
    for i in range(c):
        if(a == i):
            s+= '1'
        elif(b == i):
            s+= '1'
        else:
            s+= '0'
    l.append(s)

# odd numbers 
 
b = 1
for x in range(3, 32, 2): 

    a = x

    c = a
    print('test', a,' ',b)
    l = []
    for i in range(a):
        for ii in range(b):
            set(i,(ii + i + 1) % a,c,l) 

    #for i in l:
    #    print(i)

    s = a * b 
    for i in range(s): 
        for ii in range(s): 
            if(l[i]== l[ii] and i != ii):
                print("foud repeat :( ")
                exit(-1)
    b += 1 


# pair numbers 
#
#// 6, 5
#// 3, 5
#// 5, 3 ? 
#// set(1, 1+1), set(1, 1+2), set(1, 1+3), set(1, 1+4), set(1, 1+5) // 110000 101000 010100 010010 010001 
#// set(2, 2+1), set(2, 2+2), set(2, 2+3), set(2, 2+4), set(2, 2+5) // 011000 010100 001010 001001 101000 
#// set(3, 3+1), set(3, 3+2), set(3, 3+3), set(3, 3+4), set(3, 3+5) //  
#
#// //110000 101000 100100 
#// //011000 010100 010010
#// //001100 001010 001001
#// //000110 000101 100100
#// //000011 100010 010010


# 110000 101000 100100 100010 100001
#  1 2    1 3    1 4    1 5    1 6 
# 011000 010100 010010 010001 001010
# 2 3     2 4     2 5   2 6    3  5
# 001100  001001 001001    
# 3 4     3 5     3 6   3 6 

for x in range(3, 32, 2): 
    a = int( (x + 1)/2 )
    b = x

    c = a
    print('test', a,' ',b)
    l = []
    for i in range(a):
        for ii in range(b):
            set(i,(ii + i + 1) % a,c,l) 

    #for i in l:
    #    print(i)

    s = a * b 
    for i in range(s): 
        for ii in range(s): 
            pass
           # if(l[i]== l[ii] and i != ii):
           #     print("foud repeat :( ")
           #     exit(-1) 

   
