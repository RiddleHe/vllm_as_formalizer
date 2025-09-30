(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        orange - block
        yellow - block
        green - block
        blue - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (clear red)
        (ontable orange)
        (clear orange)
        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        (ontable blue)
        (clear blue)
        (ontable purple)
        (clear purple)
        ; red
        ; orange
        ; yellow
        ; green
        ; blue
        ; purple
    )
    (:goal (and (on yellow orange)
        (on blue yellow)
        (on purple blue)
        (on green red)))
)
    