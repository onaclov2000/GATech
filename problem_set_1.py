# The function localize takes the following arguments:
#
# colors:
#        2D list, each entry either 'R' (for red cell) or 'G' (for green cell)
#
# measurements:
#        list of measurements taken by the robot, each entry either 'R' or 'G'
#
# motions:
#        list of actions taken by the robot, each entry of the form [dy,dx],
#        where dx refers to the change in the x-direction (positive meaning
#        movement to the right) and dy refers to the change in the y-direction
#        (positive meaning movement downward)
#        NOTE: the *first* coordinate is change in y; the *second* coordinate is
#              change in x
#
# sensor_right:
#        float between 0 and 1, giving the probability that any given
#        measurement is correct; the probability that the measurement is
#        incorrect is 1-sensor_right
#
# p_move:
#        float between 0 and 1, giving the probability that any given movement
#        command takes place; the probability that the movement command fails
#        (and the robot remains still) is 1-p_move; the robot will NOT overshoot
#        its destination in this exercise
#
# The function should RETURN (not just show or print) a 2D list (of the same
# dimensions as colors) that gives the probabilities that the robot occupies
# each cell in the world.
#
# Compute the probabilities by assuming the robot initially has a uniform
# probability of being in any cell.
#
# Also assume that at each step, the robot:
# 1) first makes a movement,
# 2) then takes a measurement.
#
# Motion:
#  [0,0] - stay
#  [0,1] - right
#  [0,-1] - left
#  [1,0] - down
#  [-1,0] - up

def localize(colors,measurements,motions,sensor_right,p_move):
    # initializes p to a uniform distribution over a grid of the same dimensions as colors
    pinit = 1.0 / float(len(colors)) / float(len(colors[0]))
    p = [[pinit for row in range(len(colors[0]))] for col in range(len(colors))]
#     >>> Insert your code here <<<
    for i in range(len(motions)):
        p = move(p, motions[i], p_move)
        p = sense(p,measurements[i], sensor_right)
        
    return p

def sense(p, Z, sensor_right):
    column_sum = 0.0
    row_sum = 0.0
    sensor_wrong = 1.0 - sensor_right
    q = []
    for i in range(len(p)):
        s = []
        row_sum = 0.0
        for x in range(len(p[i])):
           if Z == colors[i][x]:
              s.append(p[i][x] * sensor_right)
           else:
              s.append(p[i][x] * sensor_wrong)
        row_sum = sum(s)
        column_sum = column_sum + row_sum
        q.append(s)
    for i in range(len(q)):
        for x in range(len(q[i])):
            q[i][x] = q[i][x] / column_sum
    return q

def move(p, U, p_move):
    q = []
    p_stay = 1.0 - p_move
    #  [0,1] - right
    for i in range(len(p)):
       s = []
       for x in range(len(p[i])):
          tmp = p_move * p[i-U[0] % len(p)][x-U[1] % len(p[i])]
          tmp = tmp + p_stay * p[i % len(p)][x % len(p[i])]
          s.append(tmp)
       q.append(s)
    return q


def show(p):
    rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x),r)) + ']' for r in p]
    print '[' + ',\n '.join(rows) + ']'
    
#############################################################
# For the following test case, your output should be 
# [[0.01105, 0.02464, 0.06799, 0.04472, 0.02465],
#  [0.00715, 0.01017, 0.08696, 0.07988, 0.00935],
#  [0.00739, 0.00894, 0.11272, 0.35350, 0.04065],
#  [0.00910, 0.00715, 0.01434, 0.04313, 0.03642]]
# (within a tolerance of +/- 0.001 for each entry)

colors = [['R','G','G','R','R'],
          ['R','R','G','R','R'],
          ['R','R','G','G','R'],
          ['R','R','R','R','R']]
measurements = ['G','G','G','G','G']
motions = [[0,0],[0,1],[1,0],[1,0],[0,1]]
p = localize(colors,measurements,motions,sensor_right = 0.7, p_move = 0.8)
correct_answer = ([0.01105, 0.02464, 0.06799, 0.04472, 0.02465],
  [0.00715, 0.01017, 0.08696, 0.07988, 0.00935],
  [0.00739, 0.00894, 0.11272, 0.35350, 0.04065],
  [0.00910, 0.00715, 0.01434, 0.04313, 0.03642])
if p == correct_answer:
    print "passed test 0"
else:
    print "Failed Test 0"
    show(p)
    show(correct_answer)

# test 1
colors = [['G', 'G', 'G'],
          ['G', 'R', 'G'],
          ['G', 'G', 'G']]
measurements = ['R']
motions = [[0,0]]
sensor_right = 1.0
p_move = 1.0
p = localize(colors,measurements,motions,sensor_right,p_move)
correct_answer = (
    [[0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0]])
if p == correct_answer:
    print "passed test 1"
else:
    print "Failed Test 1"
    show(p)
    show(correct_answer)
    
# test 2
colors = [['G', 'G', 'G'],
          ['G', 'R', 'R'],
          ['G', 'G', 'G']]
measurements = ['R']
motions = [[0,0]]
sensor_right = 1.0
p_move = 1.0
p = localize(colors,measurements,motions,sensor_right,p_move)
correct_answer = (
    [[0.0, 0.0, 0.0],
     [0.0, 0.5, 0.5],
     [0.0, 0.0, 0.0]])
if p == correct_answer:
    print "passed test 2"
else:
    print "Failed Test 2"
    show(p)
    show(correct_answer)
    
# test 3
colors = [['G', 'G', 'G'],
          ['G', 'R', 'R'],
          ['G', 'G', 'G']]
measurements = ['R']
motions = [[0,0]]
sensor_right = 0.8
p_move = 1.0
p = localize(colors,measurements,motions,sensor_right,p_move)
correct_answer = (
    [[0.06666666666, 0.06666666666, 0.06666666666],
     [0.06666666666, 0.26666666666, 0.26666666666],
     [0.06666666666, 0.06666666666, 0.06666666666]])
if p == correct_answer:
    print "passed test 3"
else:
    print "Failed Test 3"
    show(p)
    show(correct_answer)
# test 4
colors = [['G', 'G', 'G'],
          ['G', 'R', 'R'],
          ['G', 'G', 'G']]
measurements = ['R', 'R']
motions = [[0,0], [0,1]]
sensor_right = 0.8
p_move = 1.0
p = localize(colors,measurements,motions,sensor_right,p_move)
correct_answer = (
    [[0.03333333333, 0.03333333333, 0.03333333333],
     [0.13333333333, 0.13333333333, 0.53333333333],
     [0.03333333333, 0.03333333333, 0.03333333333]])
if p == correct_answer:
    print "passed test 4"
else:
    print "Failed Test 4"
    show(p)
    show(correct_answer)
# test 5
colors = [['G', 'G', 'G'],
          ['G', 'R', 'R'],
          ['G', 'G', 'G']]
measurements = ['R', 'R']
motions = [[0,0], [0,1]]
sensor_right = 1.0
p_move = 1.0
p = localize(colors,measurements,motions,sensor_right,p_move)
correct_answer = (
    [[0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0]])
if p == correct_answer:
    print "passed test 5"
else:
    print "Failed Test 5"
    show(p)
    show(correct_answer)
# test 6
colors = [['G', 'G', 'G'],
          ['G', 'R', 'R'],
          ['G', 'G', 'G']]
measurements = ['R', 'R']
motions = [[0,0], [0,1]]
sensor_right = 0.8
p_move = 0.5
p = localize(colors,measurements,motions,sensor_right,p_move)
correct_answer = (
    [[0.0289855072, 0.0289855072, 0.0289855072],
     [0.0724637681, 0.2898550724, 0.4637681159],
     [0.0289855072, 0.0289855072, 0.0289855072]])
if p == correct_answer:
    print "passed test 6"
else:
    print "Failed Test 6"
    show(p)
    show(correct_answer)
# test 7
colors = [['G', 'G', 'G'],
          ['G', 'R', 'R'],
          ['G', 'G', 'G']]
measurements = ['R', 'R']
motions = [[0,0], [0,1]]
sensor_right = 1.0
p_move = 0.5
p = localize(colors,measurements,motions,sensor_right,p_move)
correct_answer = (
    [[0.0, 0.0, 0.0],
     [0.0, 0.33333333, 0.66666666],
     [0.0, 0.0, 0.0]])
if p == correct_answer:
    print "passed test 7"
else:
    print "Failed Test 7"
    show(p)
    show(correct_answer)
    
    
# pass 0 .7,.8
# pass 1 1,1
# pass 2 1,1
# pass 3 .8,1
# pass 4 .8,1
# pass 5 1,1
# pass 6 .8,.5
# pass 7 1,.5
