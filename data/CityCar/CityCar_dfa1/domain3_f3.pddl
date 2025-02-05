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
    (q1)
    (q2)
    (q3)
    (q4)
    (q5)
    (q6)
    (q7)
    (q8)
    (q9)
    (derived1)
    (derived2)
    (derived3)
    (derived4)
    (derived5)
    (derived6)
    (derived7)
    (derived8)
    (derived9)
    (derived10)
    (derived11)
    (derived12)

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

(:derived 
  (derived1)
  (and (arrived car0 junction3-1) (not (arrived car1 junction3-0)) (not (arrived car2 junction3-3)))
)

(:derived 
  (derived2)
  (or (arrived car1 junction3-0) (arrived car2 junction3-3))
)

(:derived 
  (derived3)
  (and (arrived car0 junction3-1) (arrived car1 junction3-0) (not (arrived car2 junction3-3)))
)

(:derived 
  (derived4)
  (or (arrived car2 junction3-3) (not (arrived car0 junction3-1)))
)

(:derived 
  (derived5)
  (and (arrived car0 junction3-1) (arrived car1 junction3-0) (arrived car2 junction3-3))
)

(:derived 
  (derived6)
  (or (not (arrived car0 junction3-1)) (not (arrived car1 junction3-0)))
)

(:derived 
  (derived7)
  (or (not (arrived car0 junction3-1)) (not (arrived car1 junction3-0)) (not (arrived car2 junction3-3)))
)

(:derived 
  (derived8)
  (and (at_car_jun car0 junction3-0) (not (at_car_jun car1 junction0-3)))
)

(:derived 
  (derived9)
  (and (at_car_jun car0 junction3-0) (at_car_jun car1 junction0-3) (not (at_car_jun car2 junction0-1)))
)

(:derived 
  (derived10)
  (and (at_car_jun car0 junction3-0) (at_car_jun car1 junction0-3) (at_car_jun car2 junction0-1))
)

(:derived 
  (derived11)
  (and (at_car_jun car1 junction0-3) (not (at_car_jun car2 junction0-1)))
)

(:derived 
  (derived12)
  (and (at_car_jun car1 junction0-3) (at_car_jun car2 junction0-1))
)

(:action trans
    :parameters ()
    :precondition (and (dfa))
    :effect (and
        (state) (not (dfa))
        (when (and (q1) (derived1)) (and (q3) (not (q1))))
        (when (and (q1) (derived2)) (and (q2) (not (q1))))
        (when (and (q3) (derived3)) (and (q4) (not (q3))))
        (when (and (q3) (derived4)) (and (q2) (not (q3))))
        (when (and (q4) (derived5)) (and (q5) (not (q4))))
        (when (and (q4) (derived6)) (and (q2) (not (q4))))
        (when (and (q5) (derived7)) (and (q2) (not (q5))))
        (when (and (q6) (derived8)) (and (q7) (not (q6))))
        (when (and (q6) (derived9)) (and (q8) (not (q6))))
        (when (and (q6) (derived10)) (and (q9) (not (q6))))
        (when (and (q7) (derived11)) (and (q8) (not (q7))))
        (when (and (q7) (derived12)) (and (q9) (not (q7))))
        (when (and (q8) (at_car_jun car2 junction0-1)) (and (q9) (not (q8))))
    )
)
)
