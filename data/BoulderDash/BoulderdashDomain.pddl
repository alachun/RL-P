(define (domain BoulderDash)
  (:requirements :strips :typing :adl :negative-preconditions :equality)

  (:types
    Bat Scorpion - Enemy
    Gem Player Boulder Enemy Exit - Locatable
    Cell
  )

  (:predicates
    (at ?l - Locatable ?c - Cell)
    (oriented-up ?p - Player)
    (oriented-down ?p - Player)
    (oriented-left ?p - Player)
    (oriented-right ?p - Player)
    (connected-up ?c1 ?c2 - Cell)
    (connected-down ?c1 ?c2 - Cell)
    (connected-left ?c1 ?c2 - Cell)
    (connected-right ?c1 ?c2 - Cell)
    (terrain-ground ?c - Cell)
    (terrain-wall ?c - Cell)
    (terrain-empty ?c - Cell)
    (got ?g - Gem)
    (exited-level ?p - Player)
    (dfa)
    (state)
    (q1)
    (q2)
    (q3)
    (q4)
    (q5)
  )

  (:action turn-up
    :parameters (?p - Player)
    :precondition (and
      (not (oriented-up ?p))
      (state)
    )
    :effect (and 
      (dfa)
      (not (state))
      (when
        (oriented-left ?p)
        (not (oriented-left ?p))
      )
      (when
        (oriented-right ?p)
        (not (oriented-right ?p))
      )
      (when
        (oriented-down ?p)
        (not (oriented-down ?p))
      )
      (oriented-up ?p)
    )
  )

  (:action turn-down
    :parameters (?p - Player)
    :precondition (and
      (state)
      (not (oriented-down ?p))
    )
    :effect (and 
      (dfa)
      (not (state))
      (when
        (oriented-left ?p)
        (not (oriented-left ?p))
      )
      (when
        (oriented-right ?p)
        (not (oriented-right ?p))
      )
      (when
        (oriented-up ?p)
        (not (oriented-up ?p))
      )
      (oriented-down ?p)
    )
  )

  (:action turn-left
    :parameters (?p - Player)
    :precondition (and
    (state)
      (not (oriented-left ?p))
    )
    :effect (and 
    (dfa)
      (not (state))
      (when
        (oriented-up ?p)
        (not (oriented-up ?p))
      )
      (when
        (oriented-right ?p)
        (not (oriented-right ?p))
      )
      (when
        (oriented-down ?p)
        (not (oriented-down ?p))
      )
      (oriented-left ?p)
    )
  )

  (:action turn-right
    :parameters (?p - Player)
    :precondition (and
    (state)
      (not (oriented-right ?p))
    )
    :effect (and 
    (dfa)
      (not (state))
      (when
        (oriented-left ?p)
        (not (oriented-left ?p))
      )
      (when
        (oriented-up ?p)
        (not (oriented-up ?p))
      )
      (when
        (oriented-down ?p)
        (not (oriented-down ?p))
      )
      (oriented-right ?p)
    )
  )

  (:action move-up
    :parameters (?p - Player ?c1 ?c2 - Cell)
    :precondition (and
    (state)
      (at ?p ?c1)
      (oriented-up ?p)
      (connected-up ?c1 ?c2)
      (not (exists (?b - Boulder) (at ?b ?c2)))
      (not (terrain-wall ?c2))
    )
    :effect (and
    (dfa)
      (not (state))
      (when
        (not (terrain-empty ?c2))
        (terrain-empty ?c2)
      )
      (not (at ?p ?c1))
      (at ?p ?c2)
    )
  )

  (:action move-down
    :parameters (?p - Player ?c1 ?c2 - Cell)
    :precondition (and
    (state)
      (at ?p ?c1)
      (oriented-down ?p)
      (connected-down ?c1 ?c2)
      (not (exists (?b - Boulder) (at ?b ?c2)))
      (not (terrain-wall ?c2))
    )
    :effect (and
    (dfa)
      (not (state))
      (when
        (not (terrain-empty ?c2))
        (terrain-empty ?c2)
      )
      (not (at ?p ?c1))
      (at ?p ?c2)
    )
  )

  (:action move-left
    :parameters (?p - Player ?c1 ?c2 - Cell)
    :precondition (and
    (state)
      (at ?p ?c1)
      (oriented-left ?p)
      (connected-left ?c1 ?c2)
      (not (exists (?b - Boulder) (at ?b ?c2)))
      (not (terrain-wall ?c2))
    )
    :effect (and
    (dfa)
      (not (state))
      (when
        (not (terrain-empty ?c2))
        (terrain-empty ?c2)
      )
      (not (at ?p ?c1))
      (at ?p ?c2)
    )
  )

  (:action move-right
    :parameters (?p - Player ?c1 ?c2 - Cell)
    :precondition (and
    (state)
      (at ?p ?c1)
      (oriented-right ?p)
      (connected-right ?c1 ?c2)
      (not (exists (?b - Boulder) (at ?b ?c2)))
      (not (terrain-wall ?c2))
    )
    :effect (and
    (dfa)
      (not (state))
      (when
        (not (terrain-empty ?c2))
        (terrain-empty ?c2)
      )
      (not (at ?p ?c1))
      (at ?p ?c2)
    )
  )

  (:action get-gem
    :parameters (?p - Player ?c - Cell ?g - Gem)
    :precondition (and
    (state)
      (at ?p ?c)
      (at ?g ?c)
      (not (got ?g))
    )
    :effect (and
    (dfa)
      (not (state))
      (not (at ?g ?c))
      (got ?g)
    )
  )

  (:action dig-up
    :parameters (?p - Player ?c1 ?c2 - Cell ?b - Boulder)
    :precondition (and
    (state)
      (at ?p ?c1)
      (at ?b ?c2)
      (oriented-up ?p)
      (connected-up ?c1 ?c2)
    )
    :effect (and
     (dfa)
      (not (state))
      (not (at ?b ?c2))
    )
  )

  (:action dig-down
    :parameters (?p - Player ?c1 ?c2 - Cell ?b - Boulder)
    :precondition (and
    (state)
      (at ?p ?c1)
      (at ?b ?c2)
      (oriented-down ?p)
      (connected-down ?c1 ?c2)
    )
    :effect (and
    (dfa)
      (not (state))
      (not (at ?b ?c2))
    )
  )

  (:action dig-left
    :parameters (?p - Player ?c1 ?c2 - Cell ?b - Boulder)
    :precondition (and
    (state)
      (at ?p ?c1)
      (at ?b ?c2)
      (oriented-left ?p)
      (connected-left ?c1 ?c2)
    )
    :effect (and
     (dfa)
      (not (state))
      (not (at ?b ?c2))
    )
  )

  (:action dig-right
    :parameters (?p - Player ?c1 ?c2 - Cell ?b - Boulder)
    :precondition (and
    (state)
      (at ?p ?c1)
      (at ?b ?c2)
      (oriented-right ?p)
      (connected-right ?c1 ?c2)
    )
    :effect (and
    (dfa)
      (not (state))
      (not (at ?b ?c2))
    )
  )

  (:action exit-level
    :parameters (?p - Player ?c - Cell ?e - Exit)
    :precondition (and
    (state)
      (at ?p ?c)
      (at ?e ?c)
    )
    :effect (and
    (dfa)
      (not (state))
      (not (at ?p ?c))
      (exited-level ?p)
    )
  )
    (:action trans
    :parameters ()
    :precondition (and (dfa))
    :effect (and
        (state) (not (dfa))
        (when (and (q1) (got gem1) (not (got gem2)) (not (got gem3))) (and (q3) (not (q1))))
        (when (and (q1) (or (got gem2) (got gem3))) (and (q2) (not (q1))))
        (when (and (q3) (got gem2) (not (got gem3))) (and (q4) (not (q3))))
        (when (and (q3) (got gem3) ) (and (q2) (not (q3))))
        (when (and (q4) (got gem3)) (and (q5) (not (q4))))
    )
)
  
)
