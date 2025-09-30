(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        orange - block
        purple - block
        red - block
        green - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (clear blue)
        (ontable orange)
        (clear orange)
        (ontable purple)
        (clear purple)
        (ontable red)
        (clear red)
        (ontable green)
        (clear green)
        (ontable yellow)
        (clear yellow)
        ; blue
        ; orange
        ; purple
        ; red
        ; green
        ; yellow
    )
    (:goal (and (on orange yellow)
        (on blue red)
        (on purple green)))
)
    