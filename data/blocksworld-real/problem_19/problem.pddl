(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        red - block
        green - block
        blue - block
        orange - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (clear yellow)
        (ontable red)
        (clear red)
        (ontable green)
        (on blue green)
        (on orange blue)
        (clear orange)
        (ontable purple)
        (clear purple)
        ; yellow
        ; red
        ; green -> blue -> orange
        ; purple
    )
    (:goal (and (on blue orange)
        (on red purple)
        (on green red)
        (on yellow green)))
)
    