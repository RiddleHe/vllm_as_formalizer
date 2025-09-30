(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        blue - block
        purple - block
        red - block
        yellow - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (clear orange)
        (ontable blue)
        (clear blue)
        (ontable purple)
        (clear purple)
        (ontable red)
        (clear red)
        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        ; orange
        ; blue
        ; purple
        ; red
        ; yellow
        ; green
    )
    (:goal (and (on blue purple)
        (on green yellow)
        (on red orange)))
)
    