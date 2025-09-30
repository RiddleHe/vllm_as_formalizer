(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        green - block
        red - block
        purple - block
        yellow - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (clear orange)
        (ontable green)
        (clear green)
        (ontable red)
        (clear red)
        (ontable purple)
        (clear purple)
        (ontable yellow)
        (clear yellow)
        (ontable blue)
        (clear blue)
        ; orange
        ; green
        ; red
        ; purple
        ; yellow
        ; blue
    )
    (:goal (and (on purple green)
        (on red yellow)
        (on orange blue)))
)
    