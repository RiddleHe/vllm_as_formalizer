(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        purple - block
        red - block
        orange - block
        yellow - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (clear blue)
        (ontable purple)
        (clear purple)
        (ontable red)
        (clear red)
        (ontable orange)
        (clear orange)
        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        ; blue
        ; purple
        ; red
        ; orange
        ; yellow
        ; green
    )
    (:goal (and (on yellow blue)
        (on red yellow)
        (on orange purple)
        (on green orange)))
)
    