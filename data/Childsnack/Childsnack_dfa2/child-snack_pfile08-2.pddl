; child-snack task with 13 children and 0.4 gluten factor 
; constant factor of 1.3
; random seed: 234324

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child1 child2 child3 child4 child5 child6 child7 child8 child9 child10 child11 child12 child13 - child
    bread1 bread2 bread3 bread4 bread5 bread6 bread7 bread8 bread9 bread10 bread11 bread12 bread13 - bread-portion
    content1 content2 content3 content4 content5 content6 content7 content8 content9 content10 content11 content12 content13 - content-portion
    tray1 tray2 tray3 - tray
    table1 table2 table3 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 sandw7 sandw8 sandw9 sandw10 sandw11 sandw12 sandw13 sandw14 sandw15 sandw16 sandw17 - sandwich
  )
  (:init
     (at tray1 kitchen)
     (at tray2 kitchen)
     (at tray3 kitchen)
     (at_kitchen_bread bread1)
     (at_kitchen_bread bread2)
     (at_kitchen_bread bread3)
     (at_kitchen_bread bread4)
     (at_kitchen_bread bread5)
     (at_kitchen_bread bread6)
     (at_kitchen_bread bread7)
     (at_kitchen_bread bread8)
     (at_kitchen_bread bread9)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread11)
     (at_kitchen_bread bread12)
     (at_kitchen_bread bread13)
     (at_kitchen_content content1)
     (at_kitchen_content content2)
     (at_kitchen_content content3)
     (at_kitchen_content content4)
     (at_kitchen_content content5)
     (at_kitchen_content content6)
     (at_kitchen_content content7)
     (at_kitchen_content content8)
     (at_kitchen_content content9)
     (at_kitchen_content content10)
     (at_kitchen_content content11)
     (at_kitchen_content content12)
     (at_kitchen_content content13)
     (no_gluten_bread bread3)
     (no_gluten_bread bread12)
     (no_gluten_bread bread5)
     (no_gluten_bread bread11)
     (no_gluten_bread bread1)
     (no_gluten_content content11)
     (no_gluten_content content6)
     (no_gluten_content content2)
     (no_gluten_content content10)
     (no_gluten_content content4)
     (allergic_gluten child8)
     (allergic_gluten child1)
     (allergic_gluten child12)
     (allergic_gluten child4)
     (allergic_gluten child13)
     (not_allergic_gluten child2)
     (not_allergic_gluten child3)
     (not_allergic_gluten child5)
     (not_allergic_gluten child6)
     (not_allergic_gluten child7)
     (not_allergic_gluten child9)
     (not_allergic_gluten child10)
     (not_allergic_gluten child11)
     (waiting child1 table2)
     (waiting child2 table3)
     (waiting child3 table3)
     (waiting child4 table3)
     (waiting child5 table2)
     (waiting child6 table1)
     (waiting child7 table3)
     (waiting child8 table1)
     (waiting child9 table1)
     (waiting child10 table3)
     (waiting child11 table1)
     (waiting child12 table1)
     (waiting child13 table1)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
     (notexist sandw7)
     (notexist sandw8)
     (notexist sandw9)
     (notexist sandw10)
     (notexist sandw11)
     (notexist sandw12)
     (notexist sandw13)
     (notexist sandw14)
     (notexist sandw15)
     (notexist sandw16)
     (notexist sandw17)
     (dfa)
     (q1)
  )
  (:goal
    (and
     (served child1)
     (served child2)
     (served child3)
     (served child4)
     (served child5)
     (served child6)
     (served child7)
     (served child8)
     (served child9)
     (served child10)
     (served child11)
     (served child12)
     (served child13)
     (q10)
    )
  )
)
