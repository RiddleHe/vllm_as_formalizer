(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        green - block
        blue - block
        red - block
        purple - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (clear orange)
        (ontable green)
        (clear green)
        (ontable blue)
        (on red blue)
        (clear red)
        (ontable purple)
        (clear purple)
        (ontable yellow)
        (clear yellow)
        ; orange
        ; green
        ; blue -> red
        ; purple
        ; yellow
    )
    (:goal (and (on red green)
        (on yellow purple)
        (on blue orange)))
)
    