(define (domain child-snack)
(:requirements :typing :equality)
(:types child bread-portion content-portion sandwich tray place)
(:constants kitchen - place)

(:predicates (at_kitchen_bread ?b - bread-portion)
	     (at_kitchen_content ?c - content-portion)
     	     (at_kitchen_sandwich ?s - sandwich)
     	     (no_gluten_bread ?b - bread-portion)
       	     (no_gluten_content ?c - content-portion)
      	     (ontray ?s - sandwich ?t - tray)
       	     (no_gluten_sandwich ?s - sandwich)
	     (allergic_gluten ?c - child)
     	     (not_allergic_gluten ?c - child)
	     (served ?c - child)
	     (waiting ?c - child ?p - place)
             (at ?t - tray ?p - place)
	     (notexist ?s - sandwich)
	     (dfa)
	     (state)
	     (q1)
	     (q2)
	     (q3)
	     (q4)
	     (q5)
  )

(:action make_sandwich_no_gluten 
	 :parameters (?s - sandwich ?b - bread-portion ?c - content-portion)
	 :precondition (and (at_kitchen_bread ?b)
			    (at_kitchen_content ?c)
			    (no_gluten_bread ?b)
			    (no_gluten_content ?c)
			    (notexist ?s)
			    (state))
	 :effect (and
		   (not (at_kitchen_bread ?b))
		   (not (at_kitchen_content ?c))
		   (at_kitchen_sandwich ?s)
		   (no_gluten_sandwich ?s)
                   (not (notexist ?s))
                   (dfa)
                   (not (state))
		   ))


(:action make_sandwich
	 :parameters (?s - sandwich ?b - bread-portion ?c - content-portion)
	 :precondition (and (at_kitchen_bread ?b)
			    (at_kitchen_content ?c)
                            (notexist ?s)
                            (state)
			    )
	 :effect (and
		   (not (at_kitchen_bread ?b))
		   (not (at_kitchen_content ?c))
		   (at_kitchen_sandwich ?s)
                   (not (notexist ?s))
                   (dfa)
                   (not (state))
		   ))


(:action put_on_tray
	 :parameters (?s - sandwich ?t - tray)
	 :precondition (and  (at_kitchen_sandwich ?s)
			     (at ?t kitchen)
			     (state))
	 :effect (and
		   (not (at_kitchen_sandwich ?s))
		   (ontray ?s ?t)
		   (dfa)
		   (not (state))))


(:action serve_sandwich_no_gluten
 	:parameters (?s - sandwich ?c - child ?t - tray ?p - place)
	:precondition (and
		       (allergic_gluten ?c)
		       (ontray ?s ?t)
		       (waiting ?c ?p)
		       (no_gluten_sandwich ?s)
                       (at ?t ?p)
                       (state)
		       )
	:effect (and (not (ontray ?s ?t))
		     (served ?c)
		     (dfa)
		     (not (state))))

(:action serve_sandwich
	:parameters (?s - sandwich ?c - child ?t - tray ?p - place)
	:precondition (and (not_allergic_gluten ?c)
	                   (waiting ?c ?p)
			   (ontray ?s ?t)
			   (at ?t ?p)
			   (state))
	:effect (and (not (ontray ?s ?t))
		     (served ?c)
		     (dfa)
		     (not (state))))

(:action move_tray
	 :parameters (?t - tray ?p1 ?p2 - place)
	 :precondition (and (at ?t ?p1) (state))
	 :effect (and (not (at ?t ?p1))
		      (at ?t ?p2)
		      (dfa)
		      (not (state))))

(:action trans
    :parameters ()
    :precondition (and (dfa))
    :effect (and
        (state) (not (dfa))
        (when (and (q1) (served child1) (not (served child2)) (not (served child3))) (and (q3) (not (q1))))
        (when (and (q1) (or (served child2) (served child3))) (and (q2) (not (q1))))
        (when (and (q3) (served child1) (served child2) (not (served child3))) (and (q4) (not (q3))))
        (when (and (q3) (or (served child3) (not (served child1)))) (and (q2) (not (q3))))
        (when (and (q4) (served child1) (served child2) (served child3)) (and (q5) (not (q4))))
        (when (and (q4) (or (not (served child1)) (not (served child2)))) (and (q2) (not (q4))))
        (when (and (q5) (or (not (served child1)) (not (served child2)) (not (served child3)))) (and (q2) (not (q5))))
    )
)
)

