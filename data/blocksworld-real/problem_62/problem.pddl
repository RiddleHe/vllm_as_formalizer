(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        orange - block
        yellow - block
        red - block
        purple - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (on orange green)
        (clear orange)
        (ontable yellow)
        (on red yellow)
        (clear red)
        (ontable purple)
        (on blue purple)
        (clear blue)
        ; green -> orange
        ; yellow -> red
        ; purple -> blue
    )
    (:goal (and (on yellow orange)
        (on blue yellow)
        (on red purple)
        (on green red)))
)
    