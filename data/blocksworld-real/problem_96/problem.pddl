(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        green - block
        blue - block
        purple - block
        yellow - block
        red - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (clear orange)
        (ontable green)
        (on blue green)
        (on purple blue)
        (on yellow purple)
        (on red yellow)
        (clear red)
        ; orange
        ; green -> blue -> purple -> yellow -> red
    )
    (:goal (and (on yellow blue)
        (on green orange)
        (on red green)
        (on purple red)))
)
    