(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        yellow - block
        green - block
        blue - block
        red - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (clear orange)
        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        (ontable blue)
        (clear blue)
        (ontable red)
        (clear red)
        (ontable purple)
        (clear purple)
        ; orange
        ; yellow
        ; green
        ; blue
        ; red
        ; purple
    )
    (:goal (and (on blue green)
        (on red yellow)
        (on purple orange)))
)
    