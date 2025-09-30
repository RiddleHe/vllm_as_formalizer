(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        orange - block
        red - block
        yellow - block
        green - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (clear purple)
        (ontable orange)
        (clear orange)
        (ontable red)
        (clear red)
        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        (ontable blue)
        (clear blue)
        ; purple
        ; orange
        ; red
        ; yellow
        ; green
        ; blue
    )
    (:goal (and (on green red)
        (on blue yellow)
        (on purple orange)))
)
    