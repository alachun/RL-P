import unified_planning.model.mixins
from unified_planning.shortcuts import *
up.shortcuts.get_environment().credits_stream = None
from datetime import datetime
from unified_planning.engines import PlanGenerationResultStatus

'''citycar(四个连续时序+4F)的problem问题，用于规划器调用'''

def get_citycar_problem(graph_len,car_sum,garage_sum,road_sum,garage_at_jun,car_start_garage,car_arrived_jun,place):
    car_arrived_index=[]
    for i in range(0,len(car_arrived_jun),2):
        car_arrived_index.append(car_arrived_jun[i]*graph_len+car_arrived_jun[i+1])
    place_index = []
    for i in range(0, len(place), 2):
        place_index.append(place[i] * graph_len + place[i + 1])


    # declare user types
    car=UserType("car")
    junction=UserType("junction")
    garage=UserType("garage")
    road=UserType("road")

    # declare predicates

    same_line = up.model.Fluent('same_line', BoolType(), j1=junction,j2=junction)
    diagonal = up.model.Fluent('diagonal', BoolType(), j1=junction, j2=junction)
    at_car_jun = up.model.Fluent('at_car_jun', BoolType(), c=car,j=junction)
    at_car_road=up.model.Fluent('at_car_road',BoolType(),c=car,r=road)
    starting=up.model.Fluent('starting',BoolType(),c=car,g=garage)
    arrived = up.model.Fluent('arrived', BoolType(), c=car,j=junction)
    road_connect = up.model.Fluent('road_connect', BoolType(),r=road,j1=junction,j2=junction)
    clear=up.model.Fluent('clear',BoolType(),j=junction)
    in_place=up.model.Fluent('in_place',BoolType(),r=road)
    at_garage = up.model.Fluent('at_garage', BoolType(), g=garage,j=junction)
    q1 = up.model.Fluent('q1', BoolType())
    q2 = up.model.Fluent('q2', BoolType())
    q3 = up.model.Fluent('q3', BoolType())
    q4 = up.model.Fluent('q4', BoolType())
    q5 = up.model.Fluent('q5', BoolType())
    q6 = up.model.Fluent('q6', BoolType())
    q7 = up.model.Fluent('q7', BoolType())
    q8 = up.model.Fluent('q8', BoolType())
    q9 = up.model.Fluent('q9', BoolType())
    q10 = up.model.Fluent('q10', BoolType())
    q11 = up.model.Fluent('q11', BoolType())
    state = up.model.Fluent('state', BoolType())
    dfa = up.model.Fluent('dfa', BoolType())
    trans1_trans2 = up.model.Fluent('trans1_trans2', BoolType())
    derived11 = up.model.Fluent('derived11', BoolType())
    derived12 = up.model.Fluent('derived12', BoolType())
    derived2 = up.model.Fluent('derived2', BoolType())
    derived31 = up.model.Fluent('derived31', BoolType())
    derived32 = up.model.Fluent('derived32', BoolType())
    derived4 = up.model.Fluent('derived4', BoolType())
    derived51 = up.model.Fluent('derived51', BoolType())
    derived52 = up.model.Fluent('derived52', BoolType())
    derived6 = up.model.Fluent('derived6', BoolType())
    derived7 = up.model.Fluent('derived7', BoolType())
    derived81 = up.model.Fluent('derived81', BoolType())
    derived82 = up.model.Fluent('derived82', BoolType())
    derived91 = up.model.Fluent('derived91', BoolType())
    derived92 = up.model.Fluent('derived92', BoolType())
    derived78 = up.model.Fluent('derived78', BoolType())
    derived79 = up.model.Fluent('derived79', BoolType())
    derived7101 = up.model.Fluent('derived7101', BoolType())
    derived7102 = up.model.Fluent('derived7102', BoolType())
    derived7111 = up.model.Fluent('derived7111', BoolType())
    derived7112 = up.model.Fluent('derived7112', BoolType())
    derived89 = up.model.Fluent('derived89', BoolType())
    derived810 = up.model.Fluent('derived810', BoolType())
    derived811 = up.model.Fluent('derived811', BoolType())




    #add (typed) objects to problem
    problem = up.model.Problem('citycar')
    junctions = [up.model.Object(f'junction{i}-{j}', junction) for i in range(graph_len) for j in range(graph_len)]
    cars=[up.model.Object(f'car{i}',car) for i in range(car_sum)]
    garages = [up.model.Object(f'garage{i}', garage) for i in range(garage_sum)]
    roads = [up.model.Object(f'road{i}', road) for i in range(road_sum)]
    problem.add_objects(junctions)
    problem.add_objects(cars)
    problem.add_objects(garages)
    problem.add_objects(roads)

    # specify the initial state
    # 默认初始状态时bread,content都在厨房
    #same_line+diagonal+at_car_jun+at_car_road+starting+arrived+road_connect+clear+in_place+at_garage
    problem.add_fluent(same_line, default_initial_value=False)
    for i in range(graph_len):
        for j in range(graph_len-1):
            # 添加垂直边
            problem.set_initial_value(same_line(junctions[i * graph_len + j], junctions[i * graph_len + j + 1]), True)
            problem.set_initial_value(same_line(junctions[i * graph_len + j + 1], junctions[i * graph_len + j]), True)
            # 添加水平边
            problem.set_initial_value(same_line(junctions[i + j * graph_len], junctions[i + j * graph_len + graph_len]),
                                      True)
            problem.set_initial_value(same_line(junctions[i + j * graph_len + graph_len], junctions[i + j * graph_len]),
                                      True)
    problem.add_fluent(diagonal, default_initial_value=False)
    for i in range(graph_len-1):
        for j in range(graph_len-1):
            #正对角线
            problem.set_initial_value(diagonal(junctions[i*graph_len+j],junctions[i*graph_len+j+graph_len+1]),True)
            problem.set_initial_value(diagonal(junctions[i*graph_len+j+graph_len+1], junctions[i*graph_len+j]), True)
            #反对角线
            problem.set_initial_value(
                diagonal(junctions[i*graph_len+j+1], junctions[i*graph_len+j+graph_len]), True)
            problem.set_initial_value(
                diagonal(junctions[i * graph_len + j + graph_len], junctions[i * graph_len + j + 1]), True)

    problem.add_fluent(at_car_jun, default_initial_value=False)
    problem.add_fluent(at_car_road, default_initial_value=False)

    problem.add_fluent(starting, default_initial_value=False)
    #在gym环境中每次step时设置，单独测试需要加上
    '''for index,value in car_start_garage.items():
        problem.set_initial_value(starting(cars[index],garages[value]), True)'''

    problem.add_fluent(arrived, default_initial_value=False)
    problem.add_fluent(road_connect, default_initial_value=False)

    #单独测试时初始状态下clear应该默认为True
    problem.add_fluent(clear, default_initial_value=False)

    problem.add_fluent(in_place, default_initial_value=False)
    problem.add_fluent(at_garage, default_initial_value=False)
    for key,value in garage_at_jun.items():
        problem.set_initial_value(at_garage(garages[key],junctions[value[0]*graph_len+value[1]]), True)
    problem.add_fluent(q1, default_initial_value=True)
    problem.add_fluent(q2,default_initial_value=False)
    problem.add_fluent(q3, default_initial_value=False)
    problem.add_fluent(q4, default_initial_value=False)
    problem.add_fluent(q5, default_initial_value=False)
    problem.add_fluent(q6, default_initial_value=True)
    problem.add_fluent(q7, default_initial_value=False)
    problem.add_fluent(q8, default_initial_value=False)
    problem.add_fluent(q9, default_initial_value=False)
    problem.add_fluent(q10, default_initial_value=False)
    problem.add_fluent(q11, default_initial_value=False)
    problem.add_fluent(state, default_initial_value=False)
    problem.add_fluent(dfa, default_initial_value=True)
    problem.add_fluent(trans1_trans2, default_initial_value=False)
    problem.add_fluent(derived11, default_initial_value=False)
    problem.add_fluent(derived12, default_initial_value=False)
    problem.add_fluent(derived2, default_initial_value=False)
    problem.add_fluent(derived31, default_initial_value=False)
    problem.add_fluent(derived32, default_initial_value=False)
    problem.add_fluent(derived4, default_initial_value=False)
    problem.add_fluent(derived51, default_initial_value=False)
    problem.add_fluent(derived52, default_initial_value=False)
    problem.add_fluent(derived6, default_initial_value=False)
    problem.add_fluent(derived7, default_initial_value=False)
    problem.add_fluent(derived81, default_initial_value=False)
    problem.add_fluent(derived82, default_initial_value=False)
    problem.add_fluent(derived91, default_initial_value=False)
    problem.add_fluent(derived92, default_initial_value=False)
    problem.add_fluent(derived78, default_initial_value=False)
    problem.add_fluent(derived79, default_initial_value=False)
    problem.add_fluent(derived7101, default_initial_value=False)
    problem.add_fluent(derived7102, default_initial_value=False)
    problem.add_fluent(derived7111, default_initial_value=False)
    problem.add_fluent(derived7112, default_initial_value=False)
    problem.add_fluent(derived89, default_initial_value=False)
    problem.add_fluent(derived810, default_initial_value=False)
    problem.add_fluent(derived811, default_initial_value=False)



    # add actions

    #move_car_in_road
    move_car_in_road= up.model.InstantaneousAction('move_car_in_road', j1=junction,j2=junction,c=car,r=road)
    j1=move_car_in_road.parameter('j1')
    j2 = move_car_in_road.parameter('j2')
    c= move_car_in_road.parameter('c')
    r= move_car_in_road.parameter('r')
    move_car_in_road.add_precondition(at_car_jun(c,j1))
    move_car_in_road.add_precondition(road_connect(r,j1,j2))
    move_car_in_road.add_precondition(in_place(r))
    move_car_in_road.add_precondition(state)
    move_car_in_road.add_effect(clear(j1), True)
    move_car_in_road.add_effect(at_car_road(c,r), True)
    move_car_in_road.add_effect(at_car_jun(c,j1), False)
    move_car_in_road.add_effect(state, False)
    move_car_in_road.add_effect(dfa, True)

    #没有加动作成本，需要加上
    problem.add_action(move_car_in_road)

    # move_car_out_road
    move_car_out_road = up.model.InstantaneousAction('move_car_out_road', j1=junction, j2=junction, c=car, r=road)
    j1 = move_car_out_road.parameter('j1')
    j2 = move_car_out_road.parameter('j2')
    c = move_car_out_road.parameter('c')
    r = move_car_out_road.parameter('r')
    move_car_out_road.add_precondition(at_car_road(c, r))
    move_car_out_road.add_precondition(clear(j2))
    move_car_out_road.add_precondition(road_connect(r, j1, j2))
    move_car_out_road.add_precondition(in_place(r))
    move_car_out_road.add_precondition(state)
    move_car_out_road.add_effect(at_car_jun(c,j2), True)
    move_car_out_road.add_effect(clear(j2), False)
    move_car_out_road.add_effect(at_car_road(c, r), False)
    # 没有加动作成本，需要加上
    move_car_out_road.add_effect(state, False)
    move_car_out_road.add_effect(dfa, True)
    problem.add_action(move_car_out_road)

    #car_arrived
    car_arrived=up.model.InstantaneousAction('car_arrived',j=junction,c=car)
    j=car_arrived.parameter('j')
    c=car_arrived.parameter('c')
    car_arrived.add_precondition(at_car_jun(c,j))
    car_arrived.add_precondition(state)
    car_arrived.add_effect(clear(j),True)
    car_arrived.add_effect(arrived(c,j),True)
    car_arrived.add_effect(at_car_jun(c,j), False)
    car_arrived.add_effect(state, False)
    car_arrived.add_effect(dfa, True)
    problem.add_action(car_arrived)

    # car_start
    car_start = up.model.InstantaneousAction('car_start', j=junction, c=car,g=garage)
    j = car_start.parameter('j')
    c = car_start.parameter('c')
    g = car_start.parameter('g')
    car_start.add_precondition(at_garage(g,j))
    car_start.add_precondition(starting(c,g))
    car_start.add_precondition(clear(j))
    car_start.add_precondition(state)
    car_start.add_effect(clear(j), False)
    car_start.add_effect(at_car_jun(c, j), True)
    car_start.add_effect(starting(c,g), False)
    car_start.add_effect(state, False)
    car_start.add_effect(dfa, True)
    problem.add_action(car_start)

    #build_diagonal_oneway
    build_diagonal_oneway=up.model.InstantaneousAction('build_diagonal_oneway',j1=junction,j2=junction,r=road)
    j1=build_diagonal_oneway.parameter('j1')
    j2= build_diagonal_oneway.parameter('j2')
    r= build_diagonal_oneway.parameter('r')
    build_diagonal_oneway.add_precondition(clear(j2))
    #前提条件中(not (in_place ?r1))该如何表示？
    build_diagonal_oneway.add_precondition(Not(in_place(r)))
    build_diagonal_oneway.add_precondition(diagonal(j1,j2))
    build_diagonal_oneway.add_precondition(state)
    build_diagonal_oneway.add_effect(road_connect(r,j1,j2),True)
    build_diagonal_oneway.add_effect(in_place(r),True)
    build_diagonal_oneway.add_effect(state, False)
    build_diagonal_oneway.add_effect(dfa, True)
    problem.add_action(build_diagonal_oneway)

    # build_straight_oneway
    build_straight_oneway = up.model.InstantaneousAction('build_straight_oneway', j1=junction, j2=junction, r=road)
    j1 = build_straight_oneway.parameter('j1')
    j2 = build_straight_oneway.parameter('j2')
    r = build_straight_oneway.parameter('r')
    build_straight_oneway.add_precondition(clear(j2))
    # 前提条件中(not (in_place ?r1))该如何表示？
    build_straight_oneway.add_precondition(Not(in_place(r)))
    build_straight_oneway.add_precondition(same_line(j1, j2))
    build_straight_oneway.add_precondition(state)
    build_straight_oneway.add_effect(road_connect(r, j1, j2), True)
    build_straight_oneway.add_effect(in_place(r), True)
    build_straight_oneway.add_effect(state, False)
    build_straight_oneway.add_effect(dfa, True)
    problem.add_action(build_straight_oneway)

    # build_straight_oneway
    destroy_road = up.model.InstantaneousAction('destroy_road', j1=junction, j2=junction, r=road)
    j1 = destroy_road.parameter('j1')
    j2 = destroy_road.parameter('j2')
    r = destroy_road.parameter('r')
    destroy_road.add_precondition(road_connect(r,j1,j2))
    destroy_road.add_precondition(in_place(r))
    destroy_road.add_precondition(state)
    destroy_road.add_effect(in_place(r), False)
    destroy_road.add_effect(state, False)
    destroy_road.add_effect(dfa, True)
    destroy_road.add_effect(road_connect(r, j1, j2), False)
    for i in range(car_sum):
        destroy_road.add_effect(at_car_road(cars[i], r), False, condition=And((at_car_road(cars[i], r))))
        destroy_road.add_effect(at_car_jun(cars[i], j1), True, condition=And((at_car_road(cars[i], r))))
    problem.add_action(destroy_road)


    trans = up.model.InstantaneousAction('trans')
    trans.add_precondition(dfa)
    trans.add_effect(dfa,False)
    trans.add_effect(trans1_trans2,True)

    #四个连续时序+F(a & (F(b & (F (c & (Fd)))))) 要求小车要经过某点
    trans.add_effect(derived11,True,condition=And(arrived(cars[0],junctions[car_arrived_index[0]]),
                                            Not(arrived(cars[1],junctions[car_arrived_index[1]]))))
    trans.add_effect(derived12, True, condition=And(Not(arrived(cars[2], junctions[car_arrived_index[2]])),
                                             Not(arrived(cars[3],junctions[car_arrived_index[3]]))))
    trans.add_effect(derived2,True,condition=Or(arrived(cars[1],junctions[car_arrived_index[1]]),
                                                  arrived(cars[2], junctions[car_arrived_index[2]]),
                                                  arrived(cars[3], junctions[car_arrived_index[3]])))

    trans.add_effect(derived31, True, condition=And(arrived(cars[0], junctions[car_arrived_index[0]]),
                                              arrived(cars[1], junctions[car_arrived_index[1]])))
    trans.add_effect(derived32, True, condition=And(Not(arrived(cars[2], junctions[car_arrived_index[2]])),
                                             Not(arrived(cars[3], junctions[car_arrived_index[3]]))))
    trans.add_effect(derived4, True, condition=Or(arrived(cars[2], junctions[car_arrived_index[2]]),
                                                     arrived(cars[3], junctions[car_arrived_index[3]]),
                                                     Not(arrived(cars[0], junctions[car_arrived_index[0]]))))

    trans.add_effect(derived51, True, condition=And(arrived(cars[0], junctions[car_arrived_index[0]]),
                                               arrived(cars[1], junctions[car_arrived_index[1]])))
    trans.add_effect(derived52, True, condition=And(arrived(cars[2], junctions[car_arrived_index[2]]),
                                              Not(arrived(cars[3], junctions[car_arrived_index[3]]))))

    trans.add_effect(derived6, True, condition=Or(arrived(cars[3], junctions[car_arrived_index[3]]),
                                                      Not(arrived(cars[1], junctions[car_arrived_index[1]])),
                                                      Not(arrived(cars[0], junctions[car_arrived_index[0]]))))
    trans.add_effect(derived7, True, condition=Or(Not(arrived(cars[1], junctions[car_arrived_index[1]])),
                                                      Not(arrived(cars[2], junctions[car_arrived_index[2]])),
                                                      Not(arrived(cars[0], junctions[car_arrived_index[0]]))))
    trans.add_effect(derived81, True, condition=And(arrived(cars[0], junctions[car_arrived_index[0]]),
                                               arrived(cars[1], junctions[car_arrived_index[1]])))
    trans.add_effect(derived82, True, condition=And(arrived(cars[2], junctions[car_arrived_index[2]]),
                                              arrived(cars[3], junctions[car_arrived_index[3]])))

    trans.add_effect(derived91, True, condition=Or(Not(arrived(cars[1], junctions[car_arrived_index[1]])),
                                                      Not(arrived(cars[2], junctions[car_arrived_index[2]]))))
    trans.add_effect(derived92, True, condition=Or(Not(arrived(cars[0], junctions[car_arrived_index[0]])),
                                                     Not(arrived(cars[3], junctions[car_arrived_index[3]]))))
    trans.add_effect(derived78, True, condition=And(at_car_jun(cars[0], junctions[place_index[0]]),
                                               Not(at_car_jun(cars[1], junctions[place_index[1]]))))
    trans.add_effect(derived79, True, condition=And(at_car_jun(cars[0], junctions[place_index[0]]),
                                               at_car_jun(cars[1], junctions[place_index[1]]),
                                               Not(at_car_jun(cars[2], junctions[place_index[2]]))))
    trans.add_effect(derived7101, True, condition=And(at_car_jun(cars[0], junctions[place_index[0]]),
                                               at_car_jun(cars[1], junctions[place_index[1]])))
    trans.add_effect(derived7102, True, condition=And(at_car_jun(cars[2], junctions[place_index[2]]),
                                               Not(at_car_jun(cars[3], junctions[place_index[3]]))))
    trans.add_effect(derived7111, True, condition=And(at_car_jun(cars[0], junctions[place_index[0]]),
                                               at_car_jun(cars[1], junctions[place_index[1]])))
    trans.add_effect(derived7112, True, condition=And(at_car_jun(cars[2], junctions[place_index[2]]),
                                               at_car_jun(cars[3], junctions[place_index[3]])))

    trans.add_effect(derived89, True, condition=And(at_car_jun(cars[1], junctions[place_index[1]]),
                                               Not(at_car_jun(cars[2], junctions[place_index[2]]))))

    trans.add_effect(derived810, True, condition=And(at_car_jun(cars[1], junctions[place_index[1]]),
                                               at_car_jun(cars[2], junctions[place_index[2]]),
                                               Not(at_car_jun(cars[3], junctions[place_index[3]]))))
    trans.add_effect(derived811, True, condition=And(at_car_jun(cars[1], junctions[place_index[1]]),
                                               at_car_jun(cars[2], junctions[place_index[2]]),
                                               at_car_jun(cars[3], junctions[place_index[3]])))

    problem.add_action(trans)

    trans2 = up.model.InstantaneousAction('trans2')
    trans2.add_precondition(trans1_trans2)
    trans2.add_effect(trans1_trans2, False)
    trans2.add_effect(state, True)
    trans2.add_effect(q1, False, condition=And(q1, derived11,derived12))
    trans2.add_effect(q3, True, condition=And(q1, derived11,derived12))
    trans2.add_effect(q1, False, condition=And(q1, derived2))
    trans2.add_effect(q2, True, condition=And(q1, derived2))
    trans2.add_effect(q3, False, condition=And(q3, derived31,derived32))
    trans2.add_effect(q4, True, condition=And(q3, derived31,derived32))
    trans2.add_effect(q3, False, condition=And(q3, derived4))
    trans2.add_effect(q2, True,condition=And(q3, derived4))
    trans2.add_effect(q4, False, condition=And(q4, derived51,derived52))
    trans2.add_effect(q5, True, condition=And(q4, derived51,derived52))
    trans2.add_effect(q4, False, condition=And(q4, derived6))
    trans2.add_effect(q2, True,  condition=And(q4, derived6))
    trans2.add_effect(q5, False, condition=And(q5, derived7))
    trans2.add_effect(q2, True, condition=And(q5, derived7))
    trans2.add_effect(q5, False, condition=And(q5, derived81,derived82))
    trans2.add_effect(q6, True, condition=And(q5, derived81,derived82))
    trans2.add_effect(q6, False, condition=And(q6, Or(derived91,derived92)))
    trans2.add_effect(q2, True, condition=And(q6, Or(derived91,derived92)))
    trans2.add_effect(q7, False, condition=And(q7, derived78))
    trans2.add_effect(q8, True,  condition=And(q7, derived78))
    trans2.add_effect(q7, False, condition=And(q7, derived79))
    trans2.add_effect(q9, True, condition=And(q7, derived79))
    trans2.add_effect(q7, False, condition=And(q7,derived7101,derived7102))
    trans2.add_effect(q10, True, condition=And(q7,derived7101,derived7102))
    trans2.add_effect(q7, False, condition=And(q7, derived7111,derived7112))
    trans2.add_effect(q11, True, condition=And(q7, derived7111,derived7112))
    trans2.add_effect(q8, False, condition=And(q8, derived89))
    trans2.add_effect(q9, True, condition=And(q8, derived89))
    trans2.add_effect(q8, False, condition=And(q8, derived810))
    trans2.add_effect(q10, True,condition=And(q8, derived810))
    trans2.add_effect(q8, False, condition=And(q8, derived811))
    trans2.add_effect(q11, True, condition=And(q8, derived811))
    trans2.add_effect(q9, False, condition=And(q9, at_car_jun(cars[2], junctions[place_index[2]]),
                                               Not(at_car_jun(cars[3], junctions[place_index[3]]))))
    trans2.add_effect(q10, True, condition=And(q9, at_car_jun(cars[2], junctions[place_index[2]]),
                                               Not(at_car_jun(cars[3], junctions[place_index[3]]))))
    trans2.add_effect(q9, False, condition=And(q9, at_car_jun(cars[2], junctions[place_index[2]]),
                                               at_car_jun(cars[3], junctions[place_index[3]])))
    trans2.add_effect(q11, True, condition=And(q9, at_car_jun(cars[2], junctions[place_index[2]]),
                                               at_car_jun(cars[3], junctions[place_index[3]])))
    trans2.add_effect(q10, False, condition=And(q10, at_car_jun(cars[3], junctions[place_index[3]])))
    trans2.add_effect(q11, True, condition=And(q10, at_car_jun(cars[3], junctions[place_index[3]])))
    trans2.add_effect(derived11,False)
    trans2.add_effect(derived12, False)
    trans2.add_effect(derived2, False)
    trans2.add_effect(derived31, False)
    trans2.add_effect(derived32, False)
    trans2.add_effect(derived4, False)
    trans2.add_effect(derived51, False)
    trans2.add_effect(derived52, False)
    trans2.add_effect(derived6, False)
    trans2.add_effect(derived7, False)
    trans2.add_effect(derived81, False)
    trans2.add_effect(derived82, False)
    trans2.add_effect(derived91, False)
    trans2.add_effect(derived92, False)
    trans2.add_effect(derived78, False)
    trans2.add_effect(derived79, False)
    trans2.add_effect(derived7101, False)
    trans2.add_effect(derived7102, False)
    trans2.add_effect(derived7111, False)
    trans2.add_effect(derived7112, False)
    trans2.add_effect(derived89, False)
    trans2.add_effect(derived810, False)
    trans2.add_effect(derived811, False)
    problem.add_action(trans2)

    #problem.add_goal(q3)
    #problem.add_goal(served(childs[0]))
    problem.clear_quality_metrics()
    #动作成本
    problem.add_quality_metric(MinimizeActionCosts(costs={
        move_car_in_road:1,
        move_car_out_road:1,
        car_arrived:0,
        car_start:0,
        build_diagonal_oneway:30,
        build_straight_oneway:20,
        destroy_road:10,
        trans:0,
        trans2:0,
    }))

    return problem



if __name__=='__main__':
    #graph_len, car_sum, garage_sum, road_sum, garage_at_jun, car_start_garage, car_arrived_jun, error_place
    problem=get_citycar_problem(graph_len=3,car_sum=2,garage_sum=2,road_sum=5,garage_at_jun={0:[0,0],1:[0,0]},
                  car_start_garage={0:0,1:1},
                  car_arrived_jun=[2,0,2,1],
                  place=[2,1,2,2],
                 )
    pv=PlanValidator(problem_kind=problem.kind)
    metric=problem.quality_metrics[0]
    params={
        'fast_downward_search_time_limit': '180s',
        'fast_downward_search_config': 'let(hcea,cea(),lazy_greedy([hcea],preferred=[hcea]))',
    }
    planner = OneshotPlanner(name="fast-downward")
    start_time = datetime.now()
    result = planner.solve(problem)
    end_time = datetime.now()
    print(end_time - start_time)

    if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
        pv_res=pv.validate(problem,result.plan)
        # 记录规划解长度
        plan_len = len(result.plan.actions)
        print(f"cost={pv_res.metric_evaluations[metric]}")
    else:
        pass






























