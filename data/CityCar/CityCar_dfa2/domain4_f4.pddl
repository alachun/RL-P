(define (domain citycar)
(:requirements :typing :equality :negative-preconditions :action-costs :conditional-effects)
  (:types  
	car
	junction	
	garage
	road
	)

  (:predicates
    (same_line ?xy - junction ?xy2 - junction) ;; junctions in line (row)
    (diagonal ?x - junction ?y - junction ) ;; junctions in diagonal (on the map)
    (at_car_jun ?c - car ?x - junction) ;; a car is at the junction
    (at_car_road ?c - car ?x - road) ;; a car is in a road
    (starting ?c - car ?x - garage) ;; a car is in its initial position
    (arrived ?c - car ?x - junction) ;; a car arrived at destination
    (road_connect ?r1 - road ?xy - junction ?xy2 - junction) ;; there is a road that connects 2 junctions
    (clear ?xy - junction ) ;; the junction is clear 
    (in_place ?x - road);; the road has been put in place
    (at_garage ?g - garage ?xy - junction ) ;; position of the starting garage
    (dfa)
    (state)
    (trans1_2)
    (q1)
    (q2)
    (q3)
    (q4)
    (q5)
    (q6)
    (q7)
    (q8)
    (q9)
    (q10)
    (q11)
    (derived11)
    (derived12)
    (derived2)
    (derived31)
    (derived32)
    (derived4)
    (derived51)
    (derived52)
    (derived6)
    (derived7)
    (derived81)
    (derived82)
    (derived91)
    (derived92)
    (derived78)
    (derived79)
    (derived7101)
    (derived7102)
    (derived7111)
    (derived7112)
    (derived89)
    (derived810)
    (derived811)

  )
(:functions (total-cost) - number)

;; move the car in a road: no limit on the number of cars on the road
(:action move_car_in_road
  :parameters (?xy_initial - junction ?xy_final - junction ?machine - car ?r1 - road)
  :precondition (and 
		(at_car_jun ?machine ?xy_initial)
		(not (= ?xy_initial ?xy_final))
		(road_connect ?r1 ?xy_initial ?xy_final) 
		(in_place ?r1)
		(state)
		)
  :effect (and  
		(clear ?xy_initial)
		(at_car_road ?machine ?r1)
		(not (at_car_jun ?machine ?xy_initial) )
		(dfa)
		(not (state))
		(increase (total-cost) 1)
		)
)

;; move the car out of the road to a junction. Junction must be clear.
(:action move_car_out_road
  :parameters (?xy_initial - junction ?xy_final - junction ?machine - car ?r1 - road)
  :precondition (and 
		(at_car_road ?machine ?r1)
		(clear ?xy_final) 
		(not (= ?xy_initial ?xy_final))
		(road_connect ?r1 ?xy_initial ?xy_final) 
		(in_place ?r1)
		(state)
		)
  :effect (and  
		(at_car_jun ?machine ?xy_final)
		(not (clear ?xy_final))
		(not (at_car_road ?machine ?r1) )
		(dfa)
		(not (state))
		(increase (total-cost) 1)
		)
)

;; car in the final position. They are removed from the network and position is cleared.
(:action car_arrived
  :parameters (?xy_final - junction ?machine - car )
  :precondition (and 
		(at_car_jun ?machine ?xy_final)
		(state)
		)
  :effect (and  
		(clear ?xy_final)
		(arrived ?machine ?xy_final)
		(not (at_car_jun ?machine ?xy_final))
		(dfa)
		(not (state))
		)
)

;; car moved from the initial garage in the network.
(:action car_start
  :parameters (?xy_final - junction ?machine - car ?g - garage)
  :precondition (and 
		(at_garage ?g ?xy_final)
		(starting ?machine ?g)
		(clear ?xy_final)
		(state)
		)
  :effect (and  
		(not (clear ?xy_final))
		(at_car_jun ?machine ?xy_final)
		(not (starting ?machine ?g))
		(dfa)
		(not (state))
		)
)

;; build diagonal road
(:action build_diagonal_oneway
  :parameters (?xy_initial - junction ?xy_final - junction ?r1 - road)
  :precondition (and 
		(clear ?xy_final)
		(not (= ?xy_initial ?xy_final))
		(not (in_place ?r1))
		(diagonal ?xy_initial ?xy_final)
		(state) 
		)
  :effect (and  
		(road_connect ?r1 ?xy_initial ?xy_final)
		(in_place ?r1)
		(dfa)
		(not (state))
                (increase (total-cost) 30)
		)
)

;; build straight road
(:action build_straight_oneway
  :parameters (?xy_initial - junction ?xy_final - junction ?r1 - road)
  :precondition (and 
		(clear ?xy_final)
		(not (= ?xy_initial ?xy_final))
		(same_line ?xy_initial ?xy_final) 
		(not (in_place ?r1))
		(state)
		)
  :effect (and  
		(road_connect ?r1 ?xy_initial ?xy_final)
		(in_place ?r1)
		(dfa)
		(not (state))
                (increase (total-cost) 20)
		)
)

;; remove a road
(:action destroy_road
  :parameters (?xy_initial - junction ?xy_final - junction ?r1 - road)
  :precondition (and 
		(road_connect ?r1 ?xy_initial ?xy_final)
		(not (= ?xy_initial ?xy_final))
		(in_place ?r1)
		(state)
		)
  :effect (and  
		(not (in_place ?r1))
		(not (road_connect ?r1 ?xy_initial ?xy_final))
		(dfa)
		(not (state))
                (increase (total-cost) 10)
		(forall (?c1 - car)
                     (when (at_car_road ?c1 ?r1) 
			(and
			  (not (at_car_road ?c1 ?r1))
			  (at_car_jun ?c1 ?xy_initial)
			)
		      )
		   )
		)
)

(:action trans1
    :parameters ()
    :precondition (and (dfa))
    :effect (and
        (trans1_2) (not (dfa))
        (when (and (arrived car0 junction3-3) (not (arrived car1 junction3-0)))  (derived11))
        (when (and (not (arrived car2 junction3-0)) (not (arrived car3 junction3-0)))  (derived12))
        (when (or (arrived car1 junction3-0) (arrived car2 junction3-0) (arrived car3 junction3-0))  (derived2))
        (when (and (arrived car0 junction3-3) (arrived car1 junction3-0))  (derived31))
        (when (and (not (arrived car2 junction3-0)) (not (arrived car3 junction3-0)))  (derived32))
        (when (or (arrived car2 junction3-0) (not (arrived car0 junction3-3)) (arrived car3 junction3-0))  (derived4))
        (when (and (arrived car0 junction3-3) (arrived car1 junction3-0))  (derived51))
        (when (and (arrived car2 junction3-0) (not (arrived car3 junction3-0)))  (derived52))
        (when (or (not (arrived car0 junction3-3)) (not (arrived car1 junction3-0)) (arrived car3 junction3-0))  (derived6))
        (when (or (not (arrived car0 junction3-3)) (not (arrived car1 junction3-0)) (not (arrived car2 junction3-0)))  (derived7))
        (when (and (arrived car0 junction3-3) (arrived car1 junction3-0))  (derived81))  
        (when (and (arrived car2 junction3-0) (arrived car3 junction3-0))  (derived82))  
        (when (or (not (arrived car0 junction3-3)) (not (arrived car1 junction3-0)))  (derived91))      
        (when (or (not (arrived car2 junction3-0)) (not (arrived car3 junction3-0)))  (derived92))
        (when (and (at_car_jun car0 junction2-2) (not (at_car_jun car1 junction2-0)))  (derived78))
        (when (and (at_car_jun car0 junction2-2) (at_car_jun car1 junction2-0) (not (at_car_jun car2 junction3-1)))  (derived79))
        (when (and (at_car_jun car0 junction2-2) (at_car_jun car1 junction2-0))  (derived7101))
        (when (and (at_car_jun car2 junction3-1) (not (at_car_jun car3 junction0-2)))  (derived7102))
        (when (and (at_car_jun car0 junction2-2) (at_car_jun car1 junction2-0))  (derived7111))
        (when (and (at_car_jun car2 junction3-1) (at_car_jun car3 junction0-2))  (derived7112))
        (when (and (at_car_jun car1 junction2-0) (not (at_car_jun car2 junction3-1)))  (derived89))
        (when (and (at_car_jun car1 junction2-0) (at_car_jun car2 junction3-1) (not (at_car_jun car3 junction0-2)))  (derived810))
        (when (and (at_car_jun car1 junction2-0) (at_car_jun car2 junction3-1) (at_car_jun car3 junction0-2))  (derived811))        
    )
)

(:action trans2
    :parameters ()
    :precondition (and (trans1_2))
    :effect (and
        (state) (not (trans1_2))
        (when (and (q1) (derived11) (derived12)) (and (q3) (not (q1))))
        (when (and (q1) (derived2)) (and (q2) (not (q1))))
        (when (and (q3) (derived31) (derived32)) (and (q4) (not (q3))))
        (when (and (q3) (derived4)) (and (q2) (not (q3))))
        (when (and (q4) (derived51) (derived52)) (and (q5) (not (q4))))
        (when (and (q4) (derived6)) (and (q2) (not (q4))))
        (when (and (q5) (derived7)) (and (q2) (not (q5))))
        (when (and (q5) (derived81) (derived82)) (and (q6) (not (q5))))
        (when (and (q6) (or (derived91) (derived92))) (and (q2) (not (q6))))        
        (when (and (q7) (derived78)) (and (q8) (not (q7))))
        (when (and (q7) (derived79)) (and (q9) (not (q7))))
        (when (and (q7) (derived7101) (derived7102)) (and (q10) (not (q7))))
        (when (and (q7) (derived7111) (derived7112)) (and (q11) (not (q7))))
        (when (and (q8) (derived89)) (and (q9) (not (q8))))
        (when (and (q8) (derived810)) (and (q10) (not (q8))))
        (when (and (q8) (derived811)) (and (q11) (not (q8))))
        (when (and (q9) (at_car_jun car2 junction3-1) (not (at_car_jun car3 junction0-2))) (and (q10) (not (q9))))
        (when (and (q9) (at_car_jun car2 junction3-1) (at_car_jun car3 junction0-2)) (and (q11) (not (q9))))
        (when (and (q10) (at_car_jun car3 junction0-2)) (and (q11) (not (q10))))
        (not (derived11)) (not (derived12)) (not (derived2)) (not (derived31)) (not (derived32)) (not (derived4)) (not (derived51)) (not (derived52))
        (not (derived6)) (not (derived7)) (not (derived81)) (not (derived82)) (not (derived91)) (not (derived92)) (not (derived78)) (not (derived79))
        (not (derived7101)) (not (derived7102)) (not (derived7111)) (not (derived7112)) (not (derived89)) (not (derived810)) (not (derived811))
    )
)
)
