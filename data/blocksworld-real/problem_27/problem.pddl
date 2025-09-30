(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        blue - block
        yellow - block
        red - block
        green - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (clear purple)
        (ontable blue)
        (on yellow blue)
        (clear yellow)
        (ontable red)
        (clear red)
        (ontable green)
        (clear green)
        (ontable orange)
        (clear orange)
        ; purple
        ; blue -> yellow
        ; red
        ; green
        ; orange
    )
    (:goal (and (on orange purple)
        (on yellow orange)
        (on red yellow)
        (on green red)
        (on blue green)))
)
    