(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        red - block
        orange - block
        purple - block
        blue - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on red yellow)
        (clear red)
        (ontable orange)
        (clear orange)
        (ontable purple)
        (clear purple)
        (ontable blue)
        (clear blue)
        (ontable green)
        (clear green)
        ; yellow -> red
        ; orange
        ; purple
        ; blue
        ; green
    )
    (:goal (and (on yellow purple)
        (on blue red)
        (on orange green)))
)
    