(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        green - block
        orange - block
        purple - block
        red - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        (ontable orange)
        (clear orange)
        (ontable purple)
        (clear purple)
        (ontable red)
        (clear red)
        (ontable blue)
        (clear blue)
        ; yellow
        ; green
        ; orange
        ; purple
        ; red
        ; blue
    )
    (:goal (and (on blue orange)
        (on green red)
        (on yellow purple)))
)
    